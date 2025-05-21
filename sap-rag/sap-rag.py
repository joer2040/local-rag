# sap-rag/main.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
# sap-rag.py – Script principal para cargar PDFs y páginas web, construir el índice y hacer consultas

from process_pdfs import load_and_split_pdfs
from process_web import load_and_split_webpages
from build_index import build_faiss_index
from query_rag import query_rag

# Cargar y dividir PDFs
pdf_chunks = load_and_split_pdfs("data/")

# Cargar y dividir sitios web (añade tus URLs)
urls = [
    "https://help.sap.com/docs/intelligent-robotic-process-automation/cloud-studio-user-guide/bapi-overview?locale=es-ES"
]
web_chunks = load_and_split_webpages(urls)

# Combinar todo
all_chunks = pdf_chunks + web_chunks

# Construir el índice
texts, index, model = build_faiss_index(all_chunks)

# Hacer consulta
query = "What is a BAPI?"
response = query_rag(query, index, texts, model)
print(response)


# scripts/process_pdfs.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_and_split_pdfs(folder_path):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, filename))
            docs = loader.load()
            chunks.extend(splitter.split_documents(docs))
    return chunks


# scripts/process_web.py
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_webpages(urls):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text()
        doc = Document(page_content=text, metadata={"source": url})
        chunks.extend(splitter.split_documents([doc]))
    return chunks


# scripts/build_index.py
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


# scripts/query_rag.py
def query_rag(query, index, texts, model, top_k=3):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)
    results = [texts[i] for i in I[0]]
    context = "\n".join(results)
    prompt = f"Contexto:\n{context}\n\nPregunta: {query}\nRespuesta:"
    # Aquí deberías conectar con tu LLM local (llama.cpp, text-gen-webui, etc.)
    return prompt
