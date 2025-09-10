import faiss
import numpy as np

class Indexer:
    #-------------------------
    # FAISS indexing construction
    #------------------------
    @staticmethod
    def build_faiss(embeddings: np.ndarray):
        if embeddings.shape[0] == 0:
            dim = embeddings.shape[1] if embeddings.size else 384
            return faiss.IndexFlatIP(dim)
        # Normalize embeddings for cosine similarity (L2 norm = 1)
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index


    #-------------------------
    # Save a FAISS index to the specified file path.
    #------------------------
    @staticmethod
    def save_index(index: 'faiss.Index', path: str):
        faiss.write_index(index, path)
    #-------------------------
    # Load a FAISS index from the specified file path.
    #------------------------
    @staticmethod
    def load_index(path: str):
        return faiss.read_index(path)
