from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def build_faiss_index(chunks):
    texts = [chunk.page_content for chunk in chunks]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return texts, index, model
