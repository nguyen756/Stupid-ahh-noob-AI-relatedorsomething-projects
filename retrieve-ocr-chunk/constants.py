import os
import re

# Attempt optional imports for OCR capabilities
try:
    import pytesseract
    _tesseract_available = True
except ImportError:
    _tesseract_available = False

try:
    import easyocr
    _easyocr_available = True
except ImportError:
    _easyocr_available = False


class Constants:
    """Shared configuration constants for the RAG preprocessing pipeline."""
    # Availability flags for OCR libraries
    TESSERACT_PY_AVAILABLE = _tesseract_available
    EASYOCR_AVAILABLE = _easyocr_available
# -----------------------
# defaut for cpu run, change if run with gpu
# -----------------------
    EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
    EMBED_BATCH_SIZE = 64
    CHUNK_MAX_CHARS = 1000
    CHUNK_OVERLAP = 200
    EMBED_DTYPE = "float32"

# Output file names
    FAISS_INDEX_FILE = "index.faiss"
    CHUNKS_JSONL = "chunks.jsonl"
    EMBEDDINGS_NPY = "embeddings.npy"
    PAGES_JSONL = "pages.jsonl"
    MANIFEST_JSON = "manifest.json"

# OCR settings
    USE_TESSERACT_AUTO = True       # Use Tesseract if available, default EasyOCR
    TESSERACT_LANGS = "vie"         # vie=vietnamese, eng=english, ski=skibidi,ect..
    OCR_WORKERS = max(1, min(4, (os.cpu_count() or 2) - 1))  # Number of OCR worker processes
    OCR_DPI = 100                   # Higher = higher accuracy, keep low for big pdf
    DOWNSCALE_MAX_WIDTH = 1200      # Max width (px) to downscale images before OCR
    PAGE_RENDER_BATCH = 32          # Page render batch size for memory control

    # Regex patterns for text processing
    PARA_SPLIT_RE = re.compile(r"\n{2,}")  # Split on blank lines (paragraph delimiter)
    WIKI_HEADER_RE = re.compile(r"^\s*(={2,6})\s*(.+?)\s*\1\s*$", re.M)  # Wiki section headings

    # Base URL
    WIKI_ORIGIN = "https://vi.wikipedia.org"
