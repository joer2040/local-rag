import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_webpages(urls):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for url in urls:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text()
            doc = Document(page_content=text, metadata={"source": url})
            chunks.extend(splitter.split_documents([doc]))
        except Exception as e:
            print(f"Error procesando {url}: {e}")
    return chunks
