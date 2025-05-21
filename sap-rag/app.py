# app.py – interfaz con Streamlit para tu sistema RAG
import streamlit as st
from scripts.process_pdfs import load_and_split_pdfs
from scripts.process_web import load_and_split_webpages
from scripts.build_index import build_faiss_index
from scripts.query_rag import query_rag

st.set_page_config(page_title="SAP RAG Assistant", layout="wide")
st.title("🧠 SAP RAG Assistant (local + offline)")

# Archivos PDF
with st.sidebar:
    st.header("📄 Carga de Datos")
    use_web = st.checkbox("¿Agregar URLs?")
    pdf_folder = "data/"
    st.markdown(f"**Carpeta usada**: `{pdf_folder}`")

    if use_web:
        urls_input = st.text_area("URLs (una por línea)")
        urls = [url.strip() for url in urls_input.splitlines() if url.strip()]
    else:
        urls = []

# Procesar los datos
with st.spinner("Procesando PDFs y/o URLs..."):
    pdf_chunks = load_and_split_pdfs(pdf_folder)
    web_chunks = load_and_split_webpages(urls) if urls else []
    all_chunks = pdf_chunks + web_chunks
    texts, index, model = build_faiss_index(all_chunks)

# Interfaz de consulta
st.header("💬 Haz tu pregunta sobre SAP")
query = st.text_input("Escribe tu pregunta aquí:", placeholder="¿Cómo se configura una orden de producción en SAP PP?")

if query:
    with st.spinner("Generando respuesta..."):
        answer = query_rag(query, index, texts, model)
        st.markdown("### 🧠 Respuesta:")
        st.success(answer)

# Fin del archivo
