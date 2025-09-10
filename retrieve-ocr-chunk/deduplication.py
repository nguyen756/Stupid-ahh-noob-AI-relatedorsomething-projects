class Deduplicator:
    # ---------------------
    #Remove duplicate chunks by comparing their hash values.
    # ---------------------
    @staticmethod
    def dedupe_chunks(chunks: list, existing_hashes: set = None) -> tuple:
        seen = set(existing_hashes) if existing_hashes else set()
        unique_chunks = []
        added_hashes = set()
        for chunk in chunks:
            h = chunk.get("hash")
            if h not in seen:
                seen.add(h)
                unique_chunks.append(chunk)
                added_hashes.add(h)
        return unique_chunks, added_hashes
