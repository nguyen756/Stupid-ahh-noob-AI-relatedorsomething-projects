import os
import io
import re
import json
import hashlib
import numpy as np
from tqdm import tqdm
import pymupdf
from PIL import Image
import easyocr
from sentence_transformers import SentenceTransformer
import faiss
import nltk
from nltk.tokenize import sent_tokenize
import wikipedia
import time

nltk.download("punkt", quiet=True)

# -----------------------
# Config / hyperparams
# -----------------------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64
CHUNK_MAX_CHARS = 800
CHUNK_OVERLAP = 200
EMBED_DTYPE = "float32"

FAISS_INDEX_FILE = "index.faiss"
CHUNKS_JSONL = "chunks.jsonl"
EMBEDDINGS_NPY = "embeddings.npy"
PAGES_JSONL = "pages.jsonl"  # cache raw pages per-run
MANIFEST_JSON = "manifest.json"

# -----------------------
# Utilities
# -----------------------
def sha1(text: str, n_bytes=12):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n_bytes]

def file_sha1(path, chunk_size=8 * 1024 * 1024):
    """Stream SHA1 for large files (returns full hex digest)."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def clean_ocr_text(text: str) -> str:
    """Basic cleaning for OCR/plaintext."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # join hyphenated words
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # single newlines -> space
    text = re.sub(r"\s+", " ", text).strip()      # normalize whitespace
    return text

# -----------------------
# Ingest: PDF -> pages
# -----------------------
def pdf_to_pages(pdf_bytes: bytes, ocr_reader=None, ocr_dpi=200):
    """Return list of {"page": int, "text": str, "source": "pdf" or "pdf_ocr"}"""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc, start=1):
        page_text = page.get_text().strip()
        if page_text:
            pages.append({"page": i, "text": clean_ocr_text(page_text), "source": "pdf"})
        else:
            # fallback: render page and OCR
            pix = page.get_pixmap(dpi=ocr_dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            npimg = np.array(img)
            if ocr_reader is None:
                ocr_reader = easyocr.Reader(["en", "vi"], gpu=False)
            res = ocr_reader.readtext(npimg)
            ocr_text = "\n".join([r[1] for r in res])
            pages.append({"page": i, "text": clean_ocr_text(ocr_text), "source": "pdf_ocr"})
    return pages

# -----------------------
# Ingest: Image -> page
# -----------------------
def image_to_page(image_path: str, ocr_reader=None):
    image = Image.open(image_path).convert("RGB")
    if ocr_reader is None:
        ocr_reader = easyocr.Reader(["en", "vi"], gpu=False)
    npimg = np.array(image)
    res = ocr_reader.readtext(npimg)
    ocr_text = "\n".join([r[1] for r in res])
    return {"page": 1, "text": clean_ocr_text(ocr_text), "source": "image_ocr"}

# -----------------------
# Ingest: Wikipedia pages
# -----------------------
def fetch_wikipedia_titles(titles, lang="vi"):
    """titles: list of title strings. returns list of pages"""
    wikipedia.set_lang(lang)
    pages = []
    for t in titles or []:
        try:
            page_obj = wikipedia.page(t)
            content = page_obj.content
            url = page_obj.url
            pages.append({"page": 1, "text": clean_ocr_text(content), "source": "wiki", "url": url, "title": t})
        except Exception:
            try:
                results = wikipedia.search(t, results=5)
                if results:
                    title = results[0]
                    page_obj = wikipedia.page(title)
                    content = page_obj.content
                    url = page_obj.url
                    pages.append({"page": 1, "text": clean_ocr_text(content), "source": "wiki", "url": url, "title": title})
                else:
                    print(f"[wiki] no results for {t}")
            except Exception as e2:
                print(f"[wiki] failed to fetch {t}: {e2}")
    return pages

# -----------------------
# Chunking
# -----------------------
def chunk_pages(pages, max_chars=CHUNK_MAX_CHARS, overlap_chars=CHUNK_OVERLAP):
    chunks = []
    chunk_id = 0
    for pinfo in pages:
        page_num = pinfo.get("page", 1)
        text = pinfo.get("text", "")
        if not text or len(text) < 30:
            continue
        sents = sent_tokenize(text)
        cur = ""
        for s in sents:
            if len(cur) + len(s) <= max_chars:
                cur = (cur + " " + s).strip() if cur else s
            else:
                fingerprint = sha1(cur)
                chunks.append({
                    "id": f"chunk_{chunk_id}",
                    "doc_id": pinfo.get("title", pinfo.get("source", "doc")),
                    "source": pinfo.get("source"),
                    "page": page_num,
                    "text": cur,
                    "hash": fingerprint
                })
                chunk_id += 1
                if overlap_chars > 0:
                    tail = cur[-overlap_chars:]
                    cur = (tail + " " + s).strip()
                else:
                    cur = s
        if cur:
            fingerprint = sha1(cur)
            chunks.append({
                "id": f"chunk_{chunk_id}",
                "doc_id": pinfo.get("title", pinfo.get("source", "doc")),
                "source": pinfo.get("source"),
                "page": page_num,
                "text": cur,
                "hash": fingerprint
            })
            chunk_id += 1
    return chunks

# -----------------------
# Deduplicate
# -----------------------
def dedupe_chunks(chunks, existing_hashes=None):
    """
    If existing_hashes is provided (set), remove chunks whose hash is already present.
    Returns (unique_chunks, added_hashes_set)
    """
    seen = set(existing_hashes) if existing_hashes else set()
    unique = []
    added = set()
    for c in chunks:
        h = c.get("hash")
        if h not in seen:
            seen.add(h)
            unique.append(c)
            added.add(h)
    return unique, added

# -----------------------
# Embedding batching
# -----------------------
def embed_chunks(chunks, model_name=EMBED_MODEL_NAME, batch_size=BATCH_SIZE):
    if len(chunks) == 0:
        return np.zeros((0, SentenceTransformer(model_name).get_sentence_embedding_dimension()), dtype=EMBED_DTYPE)
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        emb_batch = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(emb_batch)
    embeddings = np.vstack(embeddings).astype(EMBED_DTYPE)
    return embeddings

# -----------------------
# FAISS helpers
# -----------------------
def build_faiss(embeddings):
    if embeddings.shape[0] == 0:
        # Create an empty index with the right dimension (guess 384 if model default)
        dim = embeddings.shape[1] if embeddings.size else 384
        index = faiss.IndexFlatIP(dim)
        return index
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

# -----------------------
# Save / Load
# -----------------------
def save_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for c in items:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items

def save_manifest(path, manifest):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------
# Manifest generation
# -----------------------
def make_manifest(pdf_paths, wiki_titles, params):
    """Create manifest describing inputs & params to detect changes."""
    pdf_infos = []
    for p in sorted(pdf_paths):
        if not os.path.exists(p):
            continue
        st = os.stat(p)
        pdf_infos.append({
            "path": os.path.abspath(p),
            "name": os.path.basename(p),
            "size": st.st_size,
            "mtime": st.st_mtime,
            # store file sha1 to detect content changes robustly
            "sha1": file_sha1(p)
        })
    manifest = {
        "pdfs": pdf_infos,
        "wiki_titles": sorted(wiki_titles or []),
        "params": {
            "CHUNK_MAX_CHARS": params.get("CHUNK_MAX_CHARS", CHUNK_MAX_CHARS),
            "CHUNK_OVERLAP": params.get("CHUNK_OVERLAP", CHUNK_OVERLAP),
            "EMBED_MODEL_NAME": params.get("EMBED_MODEL_NAME", EMBED_MODEL_NAME)
        },
        "timestamp": time.time()
    }
    return manifest

def manifests_differ(old, new):
    """Return a dict describing differences between manifests."""
    if old is None:
        return {"diff": True, "reason": "no previous manifest"}
    # Compare params
    if old.get("params", {}) != new.get("params", {}):
        return {"diff": True, "reason": "params changed"}
    old_pdfs = {p["sha1"]: p for p in old.get("pdfs", [])}
    new_pdfs = {p["sha1"]: p for p in new.get("pdfs", [])}
    # if any old SHA not present in new => file removed or changed (we'll treat as major diff)
    old_sha_set = set(old_pdfs.keys())
    new_sha_set = set(new_pdfs.keys())
    removed = old_sha_set - new_sha_set
    added = new_sha_set - old_sha_set
    # wiki changes
    if set(old.get("wiki_titles", [])) != set(new.get("wiki_titles", [])):
        wiki_changed = True
    else:
        wiki_changed = False
    if removed:
        return {"diff": True, "reason": "removed_or_changed_files", "removed": list(removed)}
    if added or wiki_changed:
        return {"diff": True, "reason": "added_files_or_wiki", "added": list(added), "wiki_changed": wiki_changed}
    return {"diff": False, "reason": "no_change"}

# -----------------------
# Orchestrator
# -----------------------
def prepare_from_pdf_paths(pdf_paths, wiki_titles=None, wiki_lang="en", out_dir="output", force=False, params=None):
    """
    Main orchestrator.
    - Detects changes via manifest.json
    - If only new files added (or new wiki titles), it will perform incremental processing and append embeddings+chunks
    - If files removed/changed or params changed or force=True -> full rebuild
    """
    params = params or {}
    os.makedirs(out_dir, exist_ok=True)

    chunks_path = os.path.join(out_dir, CHUNKS_JSONL)
    emb_path = os.path.join(out_dir, EMBEDDINGS_NPY)
    faiss_path = os.path.join(out_dir, FAISS_INDEX_FILE)
    pages_path = os.path.join(out_dir, PAGES_JSONL)
    manifest_path = os.path.join(out_dir, MANIFEST_JSON)

    new_manifest = make_manifest(pdf_paths, wiki_titles, params)

    # If previous outputs exist, load manifest and compare
    old_manifest = None
    if os.path.exists(manifest_path):
        try:
            old_manifest = load_manifest(manifest_path)
        except Exception:
            old_manifest = None

    diff = manifests_differ(old_manifest, new_manifest)

    if not force and old_manifest and diff.get("diff") is False and os.path.exists(chunks_path) and os.path.exists(emb_path) and os.path.exists(faiss_path):
        print("No changes detected. Loading existing chunks/embeddings/index...")
        chunks = load_jsonl(chunks_path)
        embeddings = np.load(emb_path)
        index = faiss.read_index(faiss_path)
        return chunks, embeddings, index

    # If manifest says only added new files or wiki changed, try incremental update
    incremental_ok = (not force and old_manifest is not None and diff.get("reason") == "added_files_or_wiki" and os.path.exists(chunks_path) and os.path.exists(emb_path) and os.path.exists(faiss_path))

    ocr_reader = easyocr.Reader(["en", "vi"], gpu=False)
    all_new_pages = []

    if incremental_ok:
        print("Detected added files or wiki titles only -> performing incremental update.")
        # load existing artifacts
        existing_chunks = load_jsonl(chunks_path)
        existing_hashes = set([c["hash"] for c in existing_chunks])
        existing_embeddings = np.load(emb_path)
        index = faiss.read_index(faiss_path)

        # find added PDFs from manifest (by sha1)
        old_shas = {p["sha1"] for p in old_manifest.get("pdfs", [])}
        added_pdf_infos = [p for p in new_manifest.get("pdfs", []) if p["sha1"] not in old_shas]

        added_paths = [p["path"] for p in added_pdf_infos]
        # process added pdf files
        for p in added_paths:
            if not os.path.exists(p):
                print(f"[warn] added file not found (skipping): {p}")
                continue
            with open(p, "rb") as fh:
                pdf_bytes = fh.read()
            pages = pdf_to_pages(pdf_bytes, ocr_reader=ocr_reader)
            for pg in pages:
                pg["title"] = os.path.basename(p)
            all_new_pages.extend(pages)

        # wiki pages (if wiki_changed)
        if diff.get("wiki_changed"):
            wiki_pages = fetch_wikipedia_titles(wiki_titles, lang=wiki_lang)
            all_new_pages.extend(wiki_pages)

        print(f"New pages to process: {len(all_new_pages)}")
        if not all_new_pages:
            # nothing new besides manifest change; save manifest and return existing
            save_manifest(manifest_path, new_manifest)
            print("No new pages found. Saved manifest and exiting.")
            return existing_chunks, existing_embeddings, index

        # chunk only new pages
        new_chunks = chunk_pages(all_new_pages)
        print(f"New chunks before dedupe: {len(new_chunks)}")
        new_chunks_unique, added_hashes = dedupe_chunks(new_chunks, existing_hashes=existing_hashes)
        print(f"New chunks after dedupe (unique): {len(new_chunks_unique)}")

        # assign chunk ids continuing from existing count
        start_id = len(existing_chunks)
        for i, c in enumerate(new_chunks_unique):
            c["id"] = f"chunk_{start_id + i}"

        # embed new chunks only
        new_embeddings = embed_chunks(new_chunks_unique, model_name=params.get("EMBED_MODEL_NAME", EMBED_MODEL_NAME), batch_size=params.get("BATCH_SIZE", BATCH_SIZE))

        # normalize then add to existing index
        if new_embeddings.shape[0] > 0:
            faiss.normalize_L2(new_embeddings)
            index.add(new_embeddings)  # append to index

            # update embeddings array and chunks list
            combined_embeddings = np.vstack([existing_embeddings, new_embeddings]).astype(EMBED_DTYPE)
            combined_chunks = existing_chunks + new_chunks_unique

            # persist
            save_jsonl(chunks_path, combined_chunks)
            np.save(emb_path, combined_embeddings)
            faiss.write_index(index, faiss_path)
            # also append new pages to pages.jsonl (we rewrite whole pages file for simplicity)
            if os.path.exists(pages_path):
                prev_pages = load_jsonl(pages_path)
            else:
                prev_pages = []
            prev_pages.extend(all_new_pages)
            save_jsonl(pages_path, prev_pages)
            save_manifest(manifest_path, new_manifest)
            print(f"Appended {len(new_chunks_unique)} chunks and {new_embeddings.shape[0]} embeddings.")
            return combined_chunks, combined_embeddings, index
        else:
            # nothing to add (all dupes)
            print("All new chunks were duplicates of existing content. No embeddings added.")
            save_manifest(manifest_path, new_manifest)
            return existing_chunks, existing_embeddings, index

    # Otherwise (force or significant changes) -> full rebuild
    print("Performing full rebuild (force or changed files/params). This may take time.")
    all_pages = []
    for p in pdf_paths:
        if not os.path.exists(p):
            print(f"[warn] pdf not found (skipping): {p}")
            continue
        with open(p, "rb") as fh:
            pdf_bytes = fh.read()
        pages = pdf_to_pages(pdf_bytes, ocr_reader=ocr_reader)
        for pg in pages:
            pg["title"] = os.path.basename(p)
        all_pages.extend(pages)

    if wiki_titles:
        wiki_pages = fetch_wikipedia_titles(wiki_titles, lang=wiki_lang)
        all_pages.extend(wiki_pages)

    print(f"Total pages collected: {len(all_pages)}")
    save_jsonl(pages_path, all_pages)

    chunks = chunk_pages(all_pages)
    print(f"Chunks before dedupe: {len(chunks)}")
    chunks = dedupe_chunks(chunks)[0]  # dedupe against itself
    print(f"Chunks after dedupe: {len(chunks)}")

    embeddings = embed_chunks(chunks, model_name=params.get("EMBED_MODEL_NAME", EMBED_MODEL_NAME), batch_size=params.get("BATCH_SIZE", BATCH_SIZE))
    index = build_faiss(embeddings)

    save_jsonl(chunks_path, chunks)
    np.save(emb_path, embeddings)
    faiss.write_index(index, faiss_path)
    save_manifest(manifest_path, new_manifest)

    print(f"Saved (full rebuild): {chunks_path}, {emb_path}, {faiss_path}")
    return chunks, embeddings, index

# -----------------------
# Loader helper for RAG
# -----------------------
def load_prepared(out_dir="output"):
    chunks_path = os.path.join(out_dir, CHUNKS_JSONL)
    emb_path = os.path.join(out_dir, EMBEDDINGS_NPY)
    faiss_path = os.path.join(out_dir, FAISS_INDEX_FILE)
    if not (os.path.exists(chunks_path) and os.path.exists(emb_path) and os.path.exists(faiss_path)):
        raise FileNotFoundError("Prepared artifacts not found in out_dir")
    chunks = load_jsonl(chunks_path)
    embeddings = np.load(emb_path)
    index = faiss.read_index(faiss_path)
    return chunks, embeddings, index

# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    pdf_list = [r"d:\\College\\_hk5\\Data\\sach_do_VN-DV_c7cf2.pdf"]
    wiki_titles = ["Danh mục sách đỏ động vật Việt Nam"]
    chunks, embeddings, index = prepare_from_pdf_paths(pdf_list, wiki_titles=wiki_titles, wiki_lang="vi", out_dir="prepared_data", force=False)
    print("Done. Chunks saved to prepared_data/")
