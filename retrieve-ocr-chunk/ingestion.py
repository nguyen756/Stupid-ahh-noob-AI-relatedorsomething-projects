import os
import io
import shutil
import requests
import numpy as np
import pymupdf  # PyMuPDF library for PDF reading
from PIL import Image
from bs4 import BeautifulSoup
from urllib.parse import urlparse, unquote, quote
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from constants import Constants
from utils import Utils

class Ingestion:
    # Static variable for EasyOCR reader (one per worker process)
    _worker_easy_reader = None
    #-------------------------
    #Return the full Wikipedia URL
    #------------------------
    @staticmethod
    def page_url_from_title(title: str, lang: str = "vi") -> str:
        return f"https://{lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"


    #-------------------------
    #Fetch an HTML page and extract Wikipedia page titles from the first column of each wikitable.
    #------------------------
    @staticmethod
    def extract_first_column_titles_from_url(main_url: str, max_titles: int = None) -> list:
        headers = {"User-Agent": "Mozilla/5.0 (RAG-bot/1.0)"}
        response = requests.get(main_url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        titles = []
        tables = soup.select("table.wikitable")
        if not tables:
            return titles
        for table in tables:
            for row in table.select("tr"):
                cells = row.find_all(["th", "td"])
                if not cells:
                    continue
                first_cell = cells[0]
                a_tag = first_cell.find("a", href=True)
                if not a_tag:
                    continue
                href = a_tag["href"]
                if not href.startswith("/wiki/"):
                    continue
                if "redlink=1" in href or ("new" in (a_tag.get("class") or [])):
                    continue
                # Convert URL path to title
                path = urlparse(href).path
                title = unquote(path.split("/wiki/", 1)[-1]).replace("_", " ")
                if title:
                    titles.append(title)
                if max_titles and len(titles) >= max_titles:
                    break
            if max_titles and len(titles) >= max_titles:
                break
        # Deduplicate while preserving order
        seen = set()
        unique_titles = []
        for t in titles:
            if t not in seen:
                seen.add(t)
                unique_titles.append(t)
        return unique_titles


    #-------------------------
    #Fetch content of Wikipedia pages given their titles. Optionally include direct linked pages
    #------------------------
    @staticmethod
    def fetch_wikipedia_titles(titles: list, lang: str = "vi", include_links: bool = False,
                                link_filter=None, max_linked_pages: int = None) -> list:
        import wikipedia
        wikipedia.set_lang(lang)
        pages = []
        visited = set()
        total_linked_fetched = 0
        linked_cap = max_linked_pages if (isinstance(max_linked_pages, int) and max_linked_pages >= 0) else None

        # 1) Fetch the seed pages
        for seed in titles or []:
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
                "text": Utils.clean_ocr_text(seed_page.content),
                "source": "wiki",
                "url": seed_page.url,
                "title": seed_page.title
            })
            # Optionally fetch direct linked pages from this seed page
            if not include_links:
                continue
            links = list(getattr(seed_page, "links", []) or [])
            if link_filter:
                links = [t for t in links if link_filter(t)]
            for t in links:
                if linked_cap is not None and total_linked_fetched >= linked_cap:
                    break
                if t in visited:
                    continue
                visited.add(t)
                try:
                    lp = wikipedia.page(t)
                except Exception:
                    hits = wikipedia.search(t, results=1)
                    if not hits:
                        continue
                    try:
                        lp = wikipedia.page(hits[0])
                    except Exception:
                        continue
                pages.append({
                    "page": 1,
                    "text": Utils.clean_ocr_text(lp.content),
                    "source": "wiki",
                    "url": lp.url,
                    "title": lp.title
                })
                total_linked_fetched += 1
            if linked_cap is not None and total_linked_fetched >= linked_cap:
                # Stop adding further linked pages beyond the cap (still fetch remaining seeds)
                include_links = False
        return pages


    #-------------------------
    #Render a PDF page to a PNG image (bytes) at the specified DPI.
    #------------------------
    @staticmethod
    def render_page_to_png_bytes(page: 'pymupdf.Page', dpi: int = Constants.OCR_DPI) -> bytes:
        mat = pymupdf.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")


    #-------------------------
    # Worker process function to perform OCR on a single page image.
    #------------------------
    @staticmethod
    def _ocr_worker_png_bytes(png_bytes: bytes, use_tesseract: bool,
                               tesseract_langs: str, downscale_max_width: int):
        try:
            from io import BytesIO
            from PIL import Image as PILImage
        except Exception as e:
            return f"[ocr_error] missing PIL in worker: {e}"

        # Open image from bytes
        try:
            img = PILImage.open(io.BytesIO(png_bytes)).convert("RGB")
        except Exception as e:
            return f"[ocr_error] failed to open image: {e}"

        # Downscale image
        try:
            w, h = img.size
            if downscale_max_width and w > downscale_max_width:
                new_h = int(h * (downscale_max_width / float(w)))
                img = img.resize((downscale_max_width, new_h), PILImage.LANCZOS)
        except Exception:
            pass

        # Try OCR with pytesseract
        if use_tesseract and Constants.TESSERACT_PY_AVAILABLE:
            try:
                import pytesseract as _pt
                config = "--psm 6"
                txt = _pt.image_to_string(img, lang=tesseract_langs, config=config)
                return Utils.clean_ocr_text(txt)
            except Exception:
                # If Tesseract OCR fails, fall through to EasyOCR
                pass

        # use EasyOCR if available
        if Constants.EASYOCR_AVAILABLE:
            try:
                # Initialize EasyOCR reader once per process
                if Ingestion._worker_easy_reader is None:
                    import easyocr as _easy
                    # Use English and Vietnamese by default (adjust as needed for other languages)
                    Ingestion._worker_easy_reader = _easy.Reader(["en", "vi"], gpu=False)
                np_img = np.array(img)
                result = Ingestion._worker_easy_reader.readtext(np_img)
                text = "\n".join([r[1] for r in result])
                return Utils.clean_ocr_text(text)
            except Exception as e:
                return f"[ocr_easy_error] {e}"

        # If no OCR backend succeeded
        return "[ocr_error] no ocr backend available in worker"


    #-------------------------
    #Run OCR on multiple page images in parallel processes.
    #------------------------
    @staticmethod
    def _run_parallel_ocr(ocr_jobs: list, use_tesseract: bool,
                           tesseract_langs: str, workers: int, downscale_max_width: int) -> list:
        pages_out = []
        if not ocr_jobs:
            return pages_out

        # Prepare the worker function with fixed parameters using partial
        worker_func = partial(Ingestion._ocr_worker_png_bytes,
                              use_tesseract=use_tesseract,
                              tesseract_langs=tesseract_langs,
                              downscale_max_width=downscale_max_width)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(worker_func, png_bytes): (title, page_no) 
                       for (title, page_no, png_bytes) in ocr_jobs}
            for future in tqdm(as_completed(futures), total=len(futures), desc="OCR pages", unit="page"):
                title, page_no = futures[future]
                try:
                    text = future.result()
                except Exception as e:
                    text = f"[ocr_exception] {e}"
                pages_out.append({
                    "page": page_no,
                    "text": text if text else "",
                    "source": "pdf_ocr",
                    "title": title
                })
        # Sort results by document title and page number for consistency
        pages_out.sort(key=lambda x: (x.get("title", ""), x.get("page", 0)))
        return pages_out


    #-------------------------
    #Read a PDF file and separate its content into text pages and OCR jobs for image-only pages.
    #------------------------
    @staticmethod
    def pdf_to_pages_with_jobs(pdf_path: str, dpi: int = Constants.OCR_DPI) -> tuple:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        doc = pymupdf.open(pdf_path)
        pages_with_text = []
        ocr_jobs = []
        base_title = os.path.basename(pdf_path)
        for i, page in enumerate(doc, start=1):
            page_text = page.get_text().strip()
            if page_text:
                pages_with_text.append({
                    "page": i,
                    "text": Utils.clean_ocr_text(page_text),
                    "source": "pdf",
                    "title": base_title
                })
            else:
                png_bytes = Ingestion.render_page_to_png_bytes(page, dpi=dpi)
                ocr_jobs.append((base_title, i, png_bytes))
        return pages_with_text, ocr_jobs
