# prepare_data_cpu.py
"""
CPU-optimized preparation pipeline for large image-only PDFs.

Features:
- Uses Tesseract (pytesseract) when available (faster & much lighter on RAM than EasyOCR on CPU).
- Falls back to EasyOCR if pytesseract or binary not found.
- Lower default DPI, downscaling, and batched OCR submission to avoid memory thrash.
- Incremental manifest logic retained.
- Parallel OCR with ProcessPoolExecutor (small number of workers by default).
"""

import os
import io
import re
import json
import hashlib
import time
import shutil
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

# pip install pytesseract easyocr sentence-transformers faiss-cpu pymupdf nltk wikipedia pillow
import pymupdf
from PIL import Image
import nltk
from nltk.tokenize import sent_tokenize
import wikipedia
import faiss
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, unquote, quote
WIKI_ORIGIN = "https://vi.wikipedia.org"
# optional imports detected at runtime
try:
    import pytesseract
    TESSERACT_PY_AVAILABLE = True
except Exception:
    TESSERACT_PY_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False

# optional to detect torch GPU for sentence-transformers (we assume CPU-only here)
try:
    import torch
except Exception:
    torch = None

nltk.download("punkt", quiet=True)

# -----------------------
# Config (CPU-friendly defaults)
# -----------------------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 64           # keep modest for CPU
CHUNK_MAX_CHARS = 800
CHUNK_OVERLAP = 200
EMBED_DTYPE = "float32"

# output files
FAISS_INDEX_FILE = "index.faiss"
CHUNKS_JSONL = "chunks.jsonl"
EMBEDDINGS_NPY = "embeddings.npy"
PAGES_JSONL = "pages.jsonl"
MANIFEST_JSON = "manifest.json"

# CPU-friendly OCR tuning
USE_TESSERACT_AUTO = True       # auto-detect; if false, force EasyOCR (only if available)
TESSERACT_LANGS = "vie"     # change if you only need 'eng' to speed up
OCR_WORKERS = max(1, min(4, (os.cpu_count() or 2) - 1))  # conservative default
OCR_DPI = 100                   # lower DPI speeds rendering and OCR
DOWNSCALE_MAX_WIDTH = 1200      # downscale wide pages to this width before OCR (px)
PAGE_RENDER_BATCH = 32          # render pages to PNG in batches before sending to OCR (memory control)

# -----------------------
# Utilities
# -----------------------
def sha1(text: str, n_bytes=12):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n_bytes]

def file_sha1(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def clean_ocr_text(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # join hyphenated words
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # single newlines -> space
    text = re.sub(r"\s+", " ", text).strip()
    return text
def page_url_from_title(title, lang="vi"):
    return f"https://{lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"

def extract_first_column_titles_from_url(main_url, max_titles=None):
    """
    Parse ALL wikitables on the page, and return Wikipedia TITLES
    taken only from the FIRST column of each table (skips redlinks).
    """
    headers = {"User-Agent": "Mozilla/5.0 (RAG-bot/1.0)"}
    r = requests.get(main_url, headers=headers, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.content, "html.parser")

    titles = []
    tables = soup.select("table.wikitable")
    if not tables:
        return titles

    for table in tables:
        for row in table.select("tr"):
            cells = row.find_all(["th", "td"])
            if not cells:
                continue
            first = cells[0]
            a = first.find("a", href=True)
            if not a:
                continue

            href = a["href"]
            if not href.startswith("/wiki/"):
                continue
            if "redlink=1" in href or ("new" in (a.get("class") or [])):
                continue

            # /wiki/Vo%E1%BB%8Dc_ch%C3%A0_v%C3%A0ng → "Voọc chà vàng"
            path = urlparse(href).path
            title = unquote(path.split("/wiki/", 1)[-1]).replace("_", " ")
            if title:
                titles.append(title)

            if max_titles and len(titles) >= max_titles:
                break
        if max_titles and len(titles) >= max_titles:
            break

    # de-dup keep order
    seen = set()
    uniq = []
    for t in titles:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq
# -----------------------
# Wikipedia ingestion
# -----------------------
def fetch_wikipedia_titles(
    titles,
    lang="vi",
    include_links=False,
    link_filter=None,
    max_linked_pages=None,
):
    """
    Fetch seed Wikipedia pages, and (optionally) ONLY their direct links.
    - No recursion: does not follow links found on linked pages.
    - max_linked_pages caps the TOTAL number of direct linked pages fetched
      across all seeds (global cap).
    - link_filter(title) -> bool can keep only certain links.
    Returns list of dicts: {"page":1,"text","source":"wiki","url","title"}
    """
    import wikipedia
    wikipedia.set_lang(lang)

    pages = []
    visited = set()
    total_linked_fetched = 0
    linked_cap = max_linked_pages if isinstance(max_linked_pages, int) and max_linked_pages >= 0 else None

    # 1) Fetch seed pages
    for seed in (titles or []):
        if seed in visited:
            continue
        visited.add(seed)
        try:
            seed_page = wikipedia.page(seed)
        except Exception:
            hits = wikipedia.search(seed, results=1)
            if not hits:
                continue
            try:
                seed_page = wikipedia.page(hits[0])
            except Exception:
                continue

        pages.append({
            "page": 1,
            "text": clean_ocr_text(seed_page.content),
            "source": "wiki",
            "url": seed_page.url,
            "title": seed_page.title,
        })

        # 2) Fetch ONLY direct links from the seed (if requested)
        if not include_links:
            continue

        links = list(seed_page.links or [])
        if link_filter:
            links = [t for t in links if link_filter(t)]

        for t in links:
            if linked_cap is not None and total_linked_fetched >= linked_cap:
                break
            if t in visited:
                continue

            # mark visited (depth-1 only)
            visited.add(t)
            try:
                lp = wikipedia.page(t)
            except Exception:
                # soft fallback
                hits = wikipedia.search(t, results=1)
                if not hits:
                    continue
                try:
                    lp = wikipedia.page(hits[0])
                except Exception:
                    continue

            pages.append({
                "page": 1,
                "text": clean_ocr_text(lp.content),
                "source": "wiki",
                "url": lp.url,
                "title": lp.title,
            })
            total_linked_fetched += 1

        # if we filled the cap during this seed, stop adding more linked pages globally
        if linked_cap is not None and total_linked_fetched >= linked_cap:
            # continue to next seeds but without adding more links; seeds themselves still get fetched
            include_links = False

    return pages
# -----------------------
# PDF rendering helper (render page -> PNG bytes)
# -----------------------
def render_page_to_png_bytes(page: pymupdf.Page, dpi=OCR_DPI) -> bytes:
    mat = pymupdf.Matrix(dpi/72.0, dpi/72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")

# -----------------------
# OCR worker (module-level for ProcessPoolExecutor)
# This worker will prefer pytesseract if requested and available; otherwise uses EasyOCR.
# -----------------------
_worker_easy_reader = None  # easyocr.Reader instance per worker (when fallback)
def _ocr_worker_png_bytes(png_bytes: bytes, use_tesseract: bool, tesseract_langs: str, downscale_max_width: int):
    """
    Input: png image bytes
    Returns: plain text string extracted (may be empty).
    This runs inside a worker process.
    """
    try:
        from io import BytesIO
        from PIL import Image as PILImage
    except Exception as e:
        return f"[ocr_error] missing PIL in worker: {e}"

    # open image
    try:
        img = PILImage.open(BytesIO(png_bytes)).convert("RGB")
    except Exception as e:
        return f"[ocr_error] failed to open image: {e}"

    # downscale if wide
    try:
        w, h = img.size
        if downscale_max_width and w > downscale_max_width:
            new_h = int(h * (downscale_max_width / float(w)))
            img = img.resize((downscale_max_width, new_h), PILImage.LANCZOS)
    except Exception:
        pass

    # Try pytesseract path first if requested
    if use_tesseract and TESSERACT_PY_AVAILABLE:
        try:
            import pytesseract as _pt
            config = "--psm 6"
            txt = _pt.image_to_string(img, lang=tesseract_langs, config=config)
            return clean_ocr_text(txt)
        except Exception as e:
            # fallback to EasyOCR below
            pass

    # Fallback: EasyOCR (heavier on CPU/RAM)
    if EASYOCR_AVAILABLE:
        try:
            global _worker_easy_reader
            if _worker_easy_reader is None:
                import easyocr as _easy
                _worker_easy_reader = _easy.Reader(["en", "vi"], gpu=False)
            npimg = np.array(img)
            res = _worker_easy_reader.readtext(npimg)
            txt = "\n".join([r[1] for r in res])
            return clean_ocr_text(txt)
        except Exception as e:
            return f"[ocr_easy_error] {e}"
    # If neither works
    return "[ocr_error] no ocr backend available in worker"

# -----------------------
# PDF -> pages (collect pages with text and queue image pages)
# -----------------------
def pdf_to_pages_with_jobs(pdf_path: str, dpi=OCR_DPI) -> Tuple[List[dict], List[Tuple[str,int,bytes]]]:
    doc = pymupdf.open(pdf_path)
    pages_with_text = []
    ocr_jobs = []
    for i, page in enumerate(doc, start=1):
        page_text = page.get_text().strip()
        if page_text:
            pages_with_text.append({"page": i, "text": clean_ocr_text(page_text), "source": "pdf", "title": os.path.basename(pdf_path)})
        else:
            png_bytes = render_page_to_png_bytes(page, dpi=dpi)
            ocr_jobs.append((os.path.basename(pdf_path), i, png_bytes))
    return pages_with_text, ocr_jobs

# -----------------------
# Chunking & dedupe
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

def dedupe_chunks(chunks, existing_hashes=None):
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
# Embedding (CPU-only)
# -----------------------
def embed_chunks(chunks, model_name=EMBED_MODEL_NAME, batch_size=EMBED_BATCH_SIZE):
    if len(chunks) == 0:
        return np.zeros((0, 384), dtype=EMBED_DTYPE)
    device = "cpu"
    print(f"[embed] Using device: {device}")
    model = SentenceTransformer(model_name, device=device)
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
        dim = embeddings.shape[1] if embeddings.size else 384
        return faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

# -----------------------
# Save / Load helpers
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

def make_manifest(pdf_paths, wiki_titles, params):
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
            "sha1": file_sha1(p)
        })
    manifest = {
        "pdfs": pdf_infos,
        "wiki_titles": sorted(wiki_titles or []),
        "params": params,
        "timestamp": time.time()
    }
    return manifest

def manifests_differ(old, new):
    if old is None:
        return {"diff": True, "reason": "no previous manifest"}
    if old.get("params", {}) != new.get("params", {}):
        return {"diff": True, "reason": "params changed"}
    old_shas = {p["sha1"] for p in old.get("pdfs", [])}
    new_shas = {p["sha1"] for p in new.get("pdfs", [])}
    removed = old_shas - new_shas
    added = new_shas - old_shas
    wiki_changed = set(old.get("wiki_titles", [])) != set(new.get("wiki_titles", []))
    if removed:
        return {"diff": True, "reason": "removed_or_changed_files", "removed": list(removed)}
    if added or wiki_changed:
        return {"diff": True, "reason": "added_files_or_wiki", "added": list(added), "wiki_changed": wiki_changed}
    return {"diff": False, "reason": "no_change"}

# -----------------------
# Parallel OCR runner (batches submissions to avoid memory spike)
# -----------------------
def _run_parallel_ocr(ocr_jobs: List[Tuple[str,int,bytes]], use_tesseract: bool, tesseract_langs: str, workers: int, downscale_max_width: int):
    """
    ocr_jobs: list of (pdf_basename, page_no, png_bytes)
    Returns pages list: {"page": page_no, "text": text, "source":"pdf_ocr", "title": title}
    """
    pages_out = []
    if not ocr_jobs:
        return pages_out

    worker_call = partial(_ocr_worker_png_bytes, use_tesseract=use_tesseract, tesseract_langs=tesseract_langs, downscale_max_width=downscale_max_width)

    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = {}
        for (title, page_no, png_bytes) in ocr_jobs:
            fut = exe.submit(worker_call, png_bytes)
            futures[fut] = (title, page_no)
        for fut in tqdm(as_completed(futures), total=len(futures), desc="OCR pages", unit="page"):
            title, page_no = futures[fut]
            try:
                text = fut.result()
            except Exception as e:
                text = f"[ocr_exception] {e}"
            pages_out.append({"page": page_no, "text": text if text else "", "source": "pdf_ocr", "title": title})

    pages_out.sort(key=lambda x: (x.get("title", ""), x.get("page", 0)))
    return pages_out

# -----------------------
# Main orchestrator (keeps incremental logic)
# -----------------------
def prepare_from_pdf_paths(pdf_paths, wiki_titles=None, wiki_lang="vi", out_dir="prepared_data_cpu", force=False, params=None):
    params = params or {}
    os.makedirs(out_dir, exist_ok=True)

    chunks_path = os.path.join(out_dir, CHUNKS_JSONL)
    emb_path = os.path.join(out_dir, EMBEDDINGS_NPY)
    faiss_path = os.path.join(out_dir, FAISS_INDEX_FILE)
    pages_path = os.path.join(out_dir, PAGES_JSONL)
    manifest_path = os.path.join(out_dir, MANIFEST_JSON)

    # merge params with defaults for manifest
    manifest_params = {
        "CHUNK_MAX_CHARS": params.get("CHUNK_MAX_CHARS", CHUNK_MAX_CHARS),
        "CHUNK_OVERLAP": params.get("CHUNK_OVERLAP", CHUNK_OVERLAP),
        "EMBED_MODEL_NAME": params.get("EMBED_MODEL_NAME", EMBED_MODEL_NAME),
        "OCR_DPI": params.get("OCR_DPI", OCR_DPI),
        "DOWNSCALE_MAX_WIDTH": params.get("DOWNSCALE_MAX_WIDTH", DOWNSCALE_MAX_WIDTH),
        "OCR_WORKERS": params.get("OCR_WORKERS", OCR_WORKERS),
        "USE_TESSERACT_AUTO": params.get("USE_TESSERACT_AUTO", USE_TESSERACT_AUTO)
    }
    new_manifest = make_manifest(pdf_paths, wiki_titles, manifest_params)

    old_manifest = load_manifest(manifest_path) if os.path.exists(manifest_path) else None
    diff = manifests_differ(old_manifest, new_manifest)

    if not force and old_manifest and diff.get("diff") is False and os.path.exists(chunks_path) and os.path.exists(emb_path) and os.path.exists(faiss_path):
        print("No changes detected. Loading existing artifacts.")
        chunks = load_jsonl(chunks_path)
        embeddings = np.load(emb_path)
        index = faiss.read_index(faiss_path)
        return chunks, embeddings, index

    incremental_ok = (not force and old_manifest is not None and diff.get("reason") == "added_files_or_wiki" and os.path.exists(chunks_path) and os.path.exists(emb_path) and os.path.exists(faiss_path))

    # Collect pages and OCR jobs across pdfs
    collected_pages = []
    ocr_jobs = []  # (title, page_no, png_bytes)
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"[warn] missing pdf: {pdf_path}; skipping")
            continue
        pages_text, jobs = pdf_to_pages_with_jobs(pdf_path, dpi=manifest_params["OCR_DPI"])
        collected_pages.extend(pages_text)
        for (title, page_no, png_bytes) in jobs:
            ocr_jobs.append((title, page_no, png_bytes))

    # Wiki pages
    wiki_pages = []
    if wiki_titles:
        MAX_ANIMALS = params.get("MAX_ANIMALS", 50)  # allow override via params
        animal_titles = []
        for seed in wiki_titles:
            seed_url = page_url_from_title(seed, lang=wiki_lang)
            animal_titles.extend(
                extract_first_column_titles_from_url(seed_url, max_titles=MAX_ANIMALS)
            )

    # depth-1 fetch: visit each animal page; do not follow its links
    wiki_pages = fetch_wikipedia_titles(
        animal_titles,
        lang=wiki_lang,
        include_links=False,
        link_filter=None,
        max_linked_pages=None
    )

    tesseract_binary_available = shutil.which("tesseract") is not None
    use_tesseract = manifest_params["USE_TESSERACT_AUTO"] and TESSERACT_PY_AVAILABLE and tesseract_binary_available

    if incremental_ok:
        print("Incremental update detected: processing only new files/wiki.")
        existing_chunks = load_jsonl(chunks_path)
        existing_hashes = set([c["hash"] for c in existing_chunks])
        existing_embeddings = np.load(emb_path)
        index = faiss.read_index(faiss_path)

        new_pages_from_ocr = []
        if ocr_jobs:
            print(f"[ocr] Running OCR on {len(ocr_jobs)} pages with {manifest_params['OCR_WORKERS']} workers (CPU mode). Using Tesseract: {use_tesseract}")
            new_pages_from_ocr = _run_parallel_ocr(ocr_jobs, use_tesseract=use_tesseract, tesseract_langs=TESSERACT_LANGS, workers=manifest_params["OCR_WORKERS"], downscale_max_width=manifest_params["DOWNSCALE_MAX_WIDTH"])

        if diff.get("wiki_changed"):
            wiki_pages = fetch_wikipedia_titles(wiki_titles, lang=wiki_lang)

        all_new_pages = collected_pages + new_pages_from_ocr + wiki_pages
        if not all_new_pages:
            save_manifest(manifest_path, new_manifest)
            return existing_chunks, existing_embeddings, index

        new_chunks = chunk_pages(all_new_pages, max_chars=manifest_params["CHUNK_MAX_CHARS"], overlap_chars=manifest_params["CHUNK_OVERLAP"])
        new_chunks_unique, added_hashes = dedupe_chunks(new_chunks, existing_hashes=existing_hashes)
        start_id = len(existing_chunks)
        for i, c in enumerate(new_chunks_unique):
            c["id"] = f"chunk_{start_id + i}"

        new_embeddings = embed_chunks(new_chunks_unique, model_name=manifest_params["EMBED_MODEL_NAME"], batch_size=EMBED_BATCH_SIZE)

        if new_embeddings.shape[0] > 0:
            faiss.normalize_L2(new_embeddings)
            index.add(new_embeddings)
            combined_embeddings = np.vstack([existing_embeddings, new_embeddings]).astype(EMBED_DTYPE)
            combined_chunks = existing_chunks + new_chunks_unique
            save_jsonl(chunks_path, combined_chunks)
            np.save(emb_path, combined_embeddings)
            faiss.write_index(index, faiss_path)
            prev_pages = load_jsonl(pages_path) if os.path.exists(pages_path) else []
            prev_pages.extend(collected_pages + new_pages_from_ocr + wiki_pages)
            save_jsonl(pages_path, prev_pages)
            save_manifest(manifest_path, new_manifest)
            print(f"Appended {len(new_chunks_unique)} chunks and {new_embeddings.shape[0]} embeddings.")
            return combined_chunks, combined_embeddings, index
        else:
            save_manifest(manifest_path, new_manifest)
            print("No unique chunks found (duplicates).")
            return existing_chunks, existing_embeddings, index

    # Full rebuild
    print("Performing full rebuild (CPU-optimized).")
    print(f"[ocr] Image pages needing OCR: {len(ocr_jobs)}. Workers: {manifest_params['OCR_WORKERS']}. DPI: {manifest_params['OCR_DPI']}. Tesseract available: {use_tesseract}")
    new_pages_from_ocr = []
    if ocr_jobs:
        new_pages_from_ocr = _run_parallel_ocr(ocr_jobs, use_tesseract=use_tesseract, tesseract_langs=TESSERACT_LANGS, workers=manifest_params["OCR_WORKERS"], downscale_max_width=manifest_params["DOWNSCALE_MAX_WIDTH"])

    all_pages = collected_pages + new_pages_from_ocr + wiki_pages
    save_jsonl(pages_path, all_pages)

    chunks = chunk_pages(all_pages, max_chars=manifest_params["CHUNK_MAX_CHARS"], overlap_chars=manifest_params["CHUNK_OVERLAP"])
    chunks, _ = dedupe_chunks(chunks)
    embeddings = embed_chunks(chunks, model_name=manifest_params["EMBED_MODEL_NAME"], batch_size=EMBED_BATCH_SIZE)
    index = build_faiss(embeddings)

    save_jsonl(chunks_path, chunks)
    np.save(emb_path, embeddings)
    faiss.write_index(index, faiss_path)
    save_manifest(manifest_path, new_manifest)
    print("Full rebuild complete.")
    return chunks, embeddings, index

# -----------------------
# Loader helper for RAG
# -----------------------
def load_prepared(out_dir="prepared_data_cpu"):
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
    # Replace with your PDF(s)
    #pdf_list = [r"D:\\College\\_hk5\\Data\\2.pdf"]
    pdf_list=[]
    wiki_titles = ["Danh mục sách đỏ động vật Việt Nam"]

    params = {
        "CHUNK_MAX_CHARS": CHUNK_MAX_CHARS,
        "CHUNK_OVERLAP": CHUNK_OVERLAP,
        "EMBED_MODEL_NAME": EMBED_MODEL_NAME,
        "OCR_DPI": OCR_DPI,
        "DOWNSCALE_MAX_WIDTH": DOWNSCALE_MAX_WIDTH,
        "OCR_WORKERS": OCR_WORKERS,
        "USE_TESSERACT_AUTO": USE_TESSERACT_AUTO
    }

    chunks, embeddings, index = prepare_from_pdf_paths(pdf_list, wiki_titles=wiki_titles, wiki_lang="vi", out_dir="prepared_data_cpu", force=False, params=params)
    print("Done. Chunks saved to prepared_data_cpu/")
