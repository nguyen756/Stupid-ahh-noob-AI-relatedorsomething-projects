#"MAX_ANIMALS": 10 adjust the number for max row wiki retrieve


import os
import shutil
import numpy as np
import faiss
from constants import Constants
from utils import Utils
from ingestion import Ingestion
from chunking import Chunker
from deduplication import Deduplicator
from embedding import Embedder
from indexing import Indexer
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

nltk.download("punkt")
class Base:


    
# Main orchestration function to prepare chunks, embeddings, and FAISS index from PDFs and Wikipedia pages.
# - If the output directory already contains up-to-date data (based on manifest), it loads existing artifacts.
# - If new files or wiki titles are added, it incrementally processes only new content.
# - If there are significant changes or force=True, it performs a full rebuild.
# Returns (chunks, embeddings, index).
    


    @staticmethod
    def prepare_from_pdf_paths(pdf_paths: list, wiki_titles: list = None, wiki_lang: str = "vi", 
                                out_dir: str = "prepared_data_cpu", force: bool = False, params: dict = None):
        params = params or {}
        os.makedirs(out_dir, exist_ok=True)

        # Define file paths for outputs
        chunks_path = os.path.join(out_dir, Constants.CHUNKS_JSONL)
        emb_path = os.path.join(out_dir, Constants.EMBEDDINGS_NPY)
        faiss_path = os.path.join(out_dir, Constants.FAISS_INDEX_FILE)
        pages_path = os.path.join(out_dir, Constants.PAGES_JSONL)
        manifest_path = os.path.join(out_dir, Constants.MANIFEST_JSON)





        # Merge provided params with defaults for manifest tracking
        manifest_params = {
            "CHUNK_MAX_CHARS": params.get("CHUNK_MAX_CHARS", Constants.CHUNK_MAX_CHARS),
            "CHUNK_OVERLAP": params.get("CHUNK_OVERLAP", Constants.CHUNK_OVERLAP),
            "EMBED_MODEL_NAME": params.get("EMBED_MODEL_NAME", Constants.EMBED_MODEL_NAME),
            "OCR_DPI": params.get("OCR_DPI", Constants.OCR_DPI),
            "DOWNSCALE_MAX_WIDTH": params.get("DOWNSCALE_MAX_WIDTH", Constants.DOWNSCALE_MAX_WIDTH),
            "OCR_WORKERS": params.get("OCR_WORKERS", Constants.OCR_WORKERS),
            "USE_TESSERACT_AUTO": params.get("USE_TESSERACT_AUTO", Constants.USE_TESSERACT_AUTO),
            "CHUNKING_STRATEGY": params.get("CHUNKING_STRATEGY", "sentences"), #or "paragraph" or "sentences" or "wiki_sections"
            "MAX_ANIMALS": params.get("MAX_ANIMALS", 10)
        }





        new_manifest = Utils.make_manifest(pdf_paths, wiki_titles, manifest_params)
        old_manifest = Utils.load_manifest(manifest_path) if os.path.exists(manifest_path) else None
        diff = Utils.manifests_differ(old_manifest, new_manifest)

        # If nothing changed and artifacts exist, load them
        if not force and old_manifest and diff.get("diff") is False \
           and os.path.exists(chunks_path) and os.path.exists(emb_path) and os.path.exists(faiss_path):
            print("No changes detected. Loading existing artifacts.")
            chunks = Utils.load_jsonl(chunks_path)
            embeddings = np.load(emb_path)
            index = Indexer.load_index(faiss_path)
            return chunks, embeddings, index

        # Determine if incremental update is applicable
        incremental_ok = (not force and old_manifest is not None 
                          and diff.get("reason") == "added_files_or_wiki"
                          and os.path.exists(chunks_path) and os.path.exists(emb_path) and os.path.exists(faiss_path))

        # Prepare pdf for OCR
        collected_pages = []
        ocr_jobs = []
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                print(f"[warn] missing pdf: {pdf_path}; skipping")
                continue
            pages_text, jobs = Ingestion.pdf_to_pages_with_jobs(pdf_path, dpi=manifest_params["OCR_DPI"])
            collected_pages.extend(pages_text)
            ocr_jobs.extend(jobs)

        # Prepare wiki pages
        wiki_pages = []
        if wiki_titles:
            max_animals = params.get("MAX_ANIMALS", 10)  # limit for titles extracted from tables
            animal_titles = []
            for seed in wiki_titles:
                seed_url = Ingestion.page_url_from_title(seed, lang=wiki_lang)
                animal_titles.extend(Ingestion.extract_first_column_titles_from_url(seed_url, max_titles=max_animals))
            wiki_pages = Ingestion.fetch_wikipedia_titles(animal_titles, lang=wiki_lang, include_links=False)

        # Decide which OCR engine to use (Tesseract if available and enabled)
        tesseract_binary_available = shutil.which("tesseract") is not None
        use_tesseract = manifest_params["USE_TESSERACT_AUTO"] and Constants.TESSERACT_PY_AVAILABLE and tesseract_binary_available

        # Incremental update path
        if incremental_ok:
            print("Incremental update detected: processing only new files/wiki.")
            existing_chunks = Utils.load_jsonl(chunks_path)
            existing_hashes = {c["hash"] for c in existing_chunks}
            existing_embeddings = np.load(emb_path)
            index = Indexer.load_index(faiss_path)

            # Perform OCR on new image pages if any
            new_pages_from_ocr = []
            if ocr_jobs:
                print(f"[ocr] Running OCR on {len(ocr_jobs)} pages with {manifest_params['OCR_WORKERS']} workers (CPU mode). Using Tesseract: {use_tesseract}")
                new_pages_from_ocr = Ingestion._run_parallel_ocr(ocr_jobs,
                                                                 use_tesseract=use_tesseract,
                                                                 tesseract_langs=Constants.TESSERACT_LANGS,
                                                                 workers=manifest_params["OCR_WORKERS"],
                                                                 downscale_max_width=manifest_params["DOWNSCALE_MAX_WIDTH"])
            # If the set of wiki seed titles changed, re-fetch those pages
            if diff.get("wiki_changed"):
                wiki_pages = Ingestion.fetch_wikipedia_titles(wiki_titles, lang=wiki_lang)

            all_new_pages = collected_pages + new_pages_from_ocr + wiki_pages
            if not all_new_pages:
                Utils.save_manifest(manifest_path, new_manifest)
                return existing_chunks, existing_embeddings, index

            new_chunks = Chunker.make_chunks(all_new_pages,
                                             strategy=manifest_params["CHUNKING_STRATEGY"],
                                             max_chars=manifest_params["CHUNK_MAX_CHARS"],
                                             overlap_chars=manifest_params["CHUNK_OVERLAP"])
            new_chunks_unique, added_hashes = Deduplicator.dedupe_chunks(new_chunks, existing_hashes=existing_hashes)
            start_id = len(existing_chunks)
            for i, chunk in enumerate(new_chunks_unique):
                chunk["id"] = f"chunk_{start_id + i}"

            new_embeddings = Embedder.embed_chunks(new_chunks_unique,
                                                   model_name=manifest_params["EMBED_MODEL_NAME"],
                                                   batch_size=Constants.EMBED_BATCH_SIZE)
            if new_embeddings.shape[0] > 0:
                faiss.normalize_L2(new_embeddings)
                index.add(new_embeddings)
                combined_embeddings = np.vstack([existing_embeddings, new_embeddings]).astype(Constants.EMBED_DTYPE)
                combined_chunks = existing_chunks + new_chunks_unique
                Utils.save_jsonl(chunks_path, combined_chunks)
                np.save(emb_path, combined_embeddings)
                Indexer.save_index(index, faiss_path)
                prev_pages = Utils.load_jsonl(pages_path) if os.path.exists(pages_path) else []
                prev_pages.extend(collected_pages + new_pages_from_ocr + wiki_pages)
                Utils.save_jsonl(pages_path, prev_pages)
                Utils.save_manifest(manifest_path, new_manifest)
                print(f"Appended {len(new_chunks_unique)} chunks and {new_embeddings.shape[0]} embeddings.")
                return combined_chunks, combined_embeddings, index
            else:
                # All new chunks were duplicates
                Utils.save_manifest(manifest_path, new_manifest)
                print("No unique chunks found (duplicates).")
                return existing_chunks, existing_embeddings, index
        print("Performing full rebuild")
        print(f"[ocr] Image pages needing OCR: {len(ocr_jobs)}. Workers: {manifest_params['OCR_WORKERS']}. "
              f"DPI: {manifest_params['OCR_DPI']}. Tesseract available: {use_tesseract}")
        new_pages_from_ocr = []
        if ocr_jobs:
            new_pages_from_ocr = Ingestion._run_parallel_ocr(ocr_jobs,
                                                             use_tesseract=use_tesseract,
                                                             tesseract_langs=Constants.TESSERACT_LANGS,
                                                             workers=manifest_params["OCR_WORKERS"],
                                                             downscale_max_width=manifest_params["DOWNSCALE_MAX_WIDTH"])
        all_pages = collected_pages + new_pages_from_ocr + wiki_pages
        Utils.save_jsonl(pages_path, all_pages)

        # Chunk all pages, remove duplicates, embed and build index
        chunks = Chunker.make_chunks(all_pages,
                                     strategy=manifest_params["CHUNKING_STRATEGY"],
                                     max_chars=manifest_params["CHUNK_MAX_CHARS"],
                                     overlap_chars=manifest_params["CHUNK_OVERLAP"])
        chunks, _ = Deduplicator.dedupe_chunks(chunks)
        embeddings = Embedder.embed_chunks(chunks,
                                           model_name=manifest_params["EMBED_MODEL_NAME"],
                                           batch_size=Constants.EMBED_BATCH_SIZE)
        index = Indexer.build_faiss(embeddings)

        Utils.save_jsonl(chunks_path, chunks)
        np.save(emb_path, embeddings)
        Indexer.save_index(index, faiss_path)
        Utils.save_manifest(manifest_path, new_manifest)
        print("Full rebuild complete.")
        return chunks, embeddings, index


    #-------------------------
    #Load previously prepared chunks, embeddings, and index from the specified output directory. Returns (chunks, embeddings, index)
    #------------------------
    @staticmethod
    def load_prepared(out_dir: str = "prepared_data_cpu"):
        chunks_path = os.path.join(out_dir, Constants.CHUNKS_JSONL)
        emb_path = os.path.join(out_dir, Constants.EMBEDDINGS_NPY)
        faiss_path = os.path.join(out_dir, Constants.FAISS_INDEX_FILE)
        if not (os.path.exists(chunks_path) and os.path.exists(emb_path) and os.path.exists(faiss_path)):
            raise FileNotFoundError("Prepared artifacts not found in out_dir")
        chunks = Utils.load_jsonl(chunks_path)
        embeddings = np.load(emb_path)
        index = Indexer.load_index(faiss_path)
        return chunks, embeddings, index

if __name__ == "__main__":
    pdf_paths = []  #[r"D:\path\to\2.pdf"]
    wiki_titles = ["Danh mục sách đỏ động vật Việt Nam"]  

    params = {
        "CHUNK_MAX_CHARS": Constants.CHUNK_MAX_CHARS,
        "CHUNK_OVERLAP": Constants.CHUNK_OVERLAP,
        "EMBED_MODEL_NAME": Constants.EMBED_MODEL_NAME,
        "OCR_DPI": Constants.OCR_DPI,
        "DOWNSCALE_MAX_WIDTH": Constants.DOWNSCALE_MAX_WIDTH,
        "OCR_WORKERS": Constants.OCR_WORKERS,
        "USE_TESSERACT_AUTO": Constants.USE_TESSERACT_AUTO,
        "CHUNKING_STRATEGY": "paragraph",  # or "sentences" | "wiki_sections" | "paragraph"
        "MAX_ANIMALS": 2, 
    }

    chunks, embeddings, index = Base.prepare_from_pdf_paths(
        pdf_paths,
        wiki_titles=wiki_titles,
        wiki_lang="vi",
        out_dir="preproccessed_data",
        force=True,
        params=params,
    )
    print(" Chunks saved in preproccessed_data")
