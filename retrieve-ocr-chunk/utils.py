import os
import re
import html
import json
import time
import hashlib
import unicodedata

class Utils:


# -----------------------
# Utilities, used to uniquely identify text chunk.
# -----------------------
    @staticmethod
    def sha1(text: str, n_bytes: int = 12) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n_bytes]


# -----------------------
# Returns the full hex digest of the file’s contents, basically detect if the pdf changes.
# -----------------------
    @staticmethod
    def file_sha1(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()


# -----------------------
# Clean OCR text 
# -----------------------
    @staticmethod
    def clean_ocr_text(text: str) -> str:
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # join hyphenated words broken by newline
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # single newlines -> space
        text = re.sub(r"\s+", " ", text).strip()      # collapse multiple spaces/newlines
        return text


# -----------------------
# Normalize vietnamese
# -----------------------
    @staticmethod
    def normalize_vi_text(s: str) -> str:
        if not s:
            return ""
        # Decode HTML entities and normalize Unicode
        s = html.unescape(s)
        s = unicodedata.normalize("NFKC", s)
        # Remove non-breaking and zero-width spaces
        s = s.replace("\u00A0", " ").replace("\u200B", "")
        # Fix hyphenation line-breaks and single newline issues
        s = re.sub(r"(\w)-\s+(\w)", r"\1\2", s)
        s = re.sub(r"(?<!\n)\n(?!\n)", " ", s)
        # Remove citation brackets like [1], [a]
        s = re.sub(r"\[\s*[0-9A-Za-z]+\s*\]", "", s)
        # Collapse whitespace
        s = re.sub(r"\s+", " ", s).strip()
        return s


# -----------------------
# Writes a list of Python objects to json
# -----------------------
    @staticmethod
    def save_jsonl(path: str, items: list):
        with open(path, "w", encoding="utf-8") as f:
            for obj in items:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# -----------------------
# Reads a JSON Lines file
# -----------------------
    @staticmethod
    def load_jsonl(path: str) -> list:
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                items.append(json.loads(line))
        return items


# -----------------------
# Save the manifest dictionary to a JSON file (pretty-printed).
# -----------------------
    @staticmethod
    def save_manifest(path: str, manifest: dict):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)


# -----------------------
# Writes a list of Python objects to json
# -----------------------
    @staticmethod
    def load_manifest(path: str) -> dict:
        """Read a manifest JSON file and return it as a dictionary."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


# -----------------------
# Create a manifest dict capturing PDF info, wiki titles, parameters, and a timestamp.
# -----------------------
    @staticmethod
    def make_manifest(pdf_paths: list, wiki_titles: list, params: dict) -> dict:
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
                "sha1": Utils.file_sha1(p)
            })
        manifest = {
            "pdfs": pdf_infos,
            "wiki_titles": sorted(wiki_titles or []),
            "params": params,
            "timestamp": time.time()
        }
        return manifest


# -----------------------
# Compare two manifest dicts and report differences.
# -----------------------
    @staticmethod
    def manifests_differ(old: dict, new: dict) -> dict:
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
