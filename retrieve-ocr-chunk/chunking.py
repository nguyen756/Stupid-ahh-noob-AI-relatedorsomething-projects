import re
from nltk.tokenize import sent_tokenize

from constants import Constants
from utils import Utils

class Chunker:

# -----------------------
# Split text into paragraphs by blank lines. Collapses 3+ consecutive newlines into 2
# -----------------------
    @staticmethod
    def _split_into_paragraphs(text: str) -> list:
        if not text:
            return []
        text = re.sub(r"\n{3,}", "\n\n", text)
        paragraphs = [p.strip() for p in re.split(Constants.PARA_SPLIT_RE, text) if p.strip()]
        return paragraphs


# -----------------------
# Split wiki text into sections
# -----------------------
    @staticmethod
    def _split_wiki_sections(text: str) -> list:
        if not text or not Constants.WIKI_HEADER_RE.search(text):
            return [("Intro", text.strip())] if text and text.strip() else []
        sections = []
        matches = list(Constants.WIKI_HEADER_RE.finditer(text))
        first = matches[0]
        intro_text = text[:first.start()].strip()
        if intro_text:
            sections.append(("Intro", intro_text))
        for i, match in enumerate(matches):
            title = match.group(2).strip()
            start = match.end()
            end = matches[i+1].start() if i+1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body:
                sections.append((title, body))
        return sections


# -----------------------
# Helper to finalize a chunk of text and append it to the chunks list.
# -----------------------
    @staticmethod
    def _emit_chunk(chunks: list, cur_text: str, chunk_id: int, pinfo: dict, extra_meta: dict = None) -> int:
        if not cur_text:
            return chunk_id
        fingerprint = Utils.sha1(cur_text)
        chunk_meta = {
            "id": f"chunk_{chunk_id}",
            "doc_id": pinfo.get("title", pinfo.get("source", "doc")),
            "source": pinfo.get("source"),
            "page": pinfo.get("page", 1),
            "text": cur_text,
            "image_url": pinfo.get("image_url"),
            "url": pinfo.get("url"),
            "hash": fingerprint
}

        if extra_meta:
            chunk_meta.update(extra_meta)
        chunks.append(chunk_meta)
        return chunk_id + 1


# -----------------------
# Pack a list of text units into chunks
# -----------------------
    @staticmethod
    def _pack_units_to_chunks(units: list, max_chars: int, overlap_chars: int,
                               chunks: list, chunk_id: int, pinfo: dict, extra_meta: dict = None) -> int:
        cur_text = ""
        for unit in units:
            if len(unit) > max_chars:
                sentences = sent_tokenize(unit)
                chunk_id = Chunker._pack_units_to_chunks(sentences, max_chars, overlap_chars, 
                                                        chunks, chunk_id, pinfo, extra_meta)
                continue
            if not cur_text:
                cur_text = unit
            elif len(cur_text) + 1 + len(unit) <= max_chars:
                cur_text = f"{cur_text} {unit}".strip()
            else:
                chunk_id = Chunker._emit_chunk(chunks, cur_text, chunk_id, pinfo, extra_meta)
                if overlap_chars > 0:
                    tail = cur_text[-overlap_chars:]
                    cur_text = f"{tail} {unit}".strip()
                else:
                    cur_text = unit
        if cur_text:
            chunk_id = Chunker._emit_chunk(chunks, cur_text, chunk_id, pinfo, extra_meta)
        return chunk_id


# -----------------------
#  Sentence-based chunking. Splits each page's text into sentences and groups them into chunks.
# -----------------------
    @staticmethod
    def chunk_pages(pages: list, max_chars: int = Constants.CHUNK_MAX_CHARS, 
     overlap_chars: int = Constants.CHUNK_OVERLAP) -> list:

        chunks = []
        chunk_id = 0
        for pinfo in pages:
            page_num = pinfo.get("page", 1)
            text = pinfo.get("text", "") or ""
            if not text or len(text) < 30:
                # Skip very short pages
                continue
            sentences = sent_tokenize(text)
            cur_text = ""
            for s in sentences:
                if len(cur_text) + len(s) <= max_chars:
                    cur_text = f"{cur_text} {s}".strip() if cur_text else s
                else:
                    chunk_id = Chunker._emit_chunk(chunks, cur_text, chunk_id, pinfo)
                    if overlap_chars > 0:
                        tail = cur_text[-overlap_chars:]
                        cur_text = f"{tail} {s}".strip()
                    else:
                        cur_text = s
            if cur_text:
                chunk_id = Chunker._emit_chunk(chunks, cur_text, chunk_id, pinfo)
        return chunks


# -----------------------
#  Paragraph-based chunking. Splits text by paragraphs (blank lines) and then packs paragraphs into chunks.
# -----------------------
    @staticmethod
    def chunk_pages_paragraph(pages: list, max_chars: int = Constants.CHUNK_MAX_CHARS, 
                               overlap_chars: int = Constants.CHUNK_OVERLAP) -> list:
        chunks = []
        chunk_id = 0
        for pinfo in pages:
            text = pinfo.get("text", "") or ""
            if not text or len(text.strip()) < 30:
                continue
            paragraphs = Chunker._split_into_paragraphs(text)
            if not paragraphs:
                continue
            chunk_id = Chunker._pack_units_to_chunks(paragraphs, max_chars, overlap_chars, 
                                                    chunks, chunk_id, pinfo)
        return chunks


# -----------------------
# Wiki section-aware chunking. For pages with source "wiki": split into sections, then chunk within each section.
# For other sources: return to paragraph chunking. Adds 'section' metadata for wiki section chunks.
# -----------------------
    @staticmethod
    def chunk_pages_wiki_sections(pages: list, max_chars: int = Constants.CHUNK_MAX_CHARS, 
                                   overlap_chars: int = Constants.CHUNK_OVERLAP) -> list:
        chunks = []
        chunk_id = 0
        for pinfo in pages:
            text = pinfo.get("text", "") or ""
            if not text or len(text.strip()) < 30:
                continue
            if pinfo.get("source") == "wiki":
                sections = Chunker._split_wiki_sections(text)
                if not sections:
                    continue
                for sec_title, body in sections:
                    paragraphs = Chunker._split_into_paragraphs(body) or [body]
                    extra_meta = {"section": sec_title}
                    chunk_id = Chunker._pack_units_to_chunks(paragraphs, max_chars, overlap_chars, 
                                                            chunks, chunk_id, pinfo, extra_meta=extra_meta)
            else:
                paragraphs = Chunker._split_into_paragraphs(text)
                if not paragraphs:
                    continue
                chunk_id = Chunker._pack_units_to_chunks(paragraphs, max_chars, overlap_chars, 
                                                        chunks, chunk_id, pinfo)
        return chunks


# -----------------------
# Create chunks from pages using the specified strategy.
# -----------------------
    @staticmethod
    def make_chunks(pages: list, strategy: str = "sentences", max_chars: int = Constants.CHUNK_MAX_CHARS, 
                     overlap_chars: int = Constants.CHUNK_OVERLAP) -> list:
        if strategy == "paragraph":
            return Chunker.chunk_pages_paragraph(pages, max_chars=max_chars, overlap_chars=overlap_chars)
        if strategy == "wiki_sections":
            return Chunker.chunk_pages_wiki_sections(pages, max_chars=max_chars, overlap_chars=overlap_chars)
        # Default
        return Chunker.chunk_pages(pages, max_chars=max_chars, overlap_chars=overlap_chars)
