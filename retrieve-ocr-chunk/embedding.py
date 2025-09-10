import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from constants import Constants
from utils import Utils

class Embedder:
    #-------------------------
    #Embed text chunks into vectors using a SentenceTransformer model."""
    #------------------------
    @staticmethod
    def embed_chunks(chunks: list, model_name: str = Constants.EMBED_MODEL_NAME, 
                      batch_size: int = Constants.EMBED_BATCH_SIZE):
        if len(chunks) == 0:
            return np.zeros((0, 384), dtype=Constants.EMBED_DTYPE)
        device = "cpu"
        print(f"[embed] Using device: {device}")
        model = SentenceTransformer(model_name, device=device)
        texts = [Utils.normalize_vi_text(chunk["text"]) for chunk in chunks]
        embeddings_list = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch_texts = texts[i:i + batch_size]
            emb_batch = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
            embeddings_list.append(emb_batch)
        embeddings = np.vstack(embeddings_list).astype(Constants.EMBED_DTYPE)
        return embeddings
