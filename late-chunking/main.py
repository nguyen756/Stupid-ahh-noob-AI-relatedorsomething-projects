import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    return tokenizer, model

tokenizer, model = load_model()

# -------------------------------
# Helper Functions
# -------------------------------

def mean_pooling(token_embeddings, attention_mask):
    """Mean pooling on token embeddings considering attention mask."""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

def late_chunking(document_text, max_tokens=128):
    """Split document into chunks based on max token length."""
    tokens = tokenizer(document_text, return_tensors='pt', truncation=False, padding=False)
    input_ids = tokens["input_ids"].squeeze(0)

    chunks = []
    for i in range(0, len(input_ids), max_tokens):
        chunk_ids = input_ids[i:i+max_tokens]
        text_chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(text_chunk)
    return chunks

def embed_texts(texts):
    """Embed a list of texts using manual mean pooling."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
    return embeddings.cpu().numpy()

def retrieve(query, chunk_texts, chunk_embeddings, top_k=3):
    """Retrieve top-k similar chunks for a query."""
    query_embedding = embed_texts([query])[0].reshape(1, -1)
    sims = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = sims.argsort()[-top_k:][::-1]
    return [(chunk_texts[i], float(sims[i])) for i in top_indices]

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📄Txt file only(currently) for late-chunking text")

uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
    st.subheader("Preview of document:")
    st.write(text[:500] + "..." if len(text) > 500 else text)

    # Process chunks
    with st.spinner("Chunking and embedding text..."):
        chunks = late_chunking(text, max_tokens=128)
        embeddings = embed_texts(chunks)

    st.success(f"Processed {len(chunks)} chunks.")
    st.write("You can now enter a query to retrieve relevant chunks. Then we will display the top 3 matches. something like \"machine learning application\"")
    query = st.text_input("Enter your search query:")
    if query:
        results = retrieve(query, chunks, embeddings, top_k=3)
        st.subheader("Top Matches:")
        for chunk, score in results:
            st.markdown(f"**Score:** {score:.4f}")
            st.write(chunk)
