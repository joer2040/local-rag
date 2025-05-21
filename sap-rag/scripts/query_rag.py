import numpy as np
import requests
import json  # Asegúrate de importar json

def query_rag(query, index, texts, model, top_k=3):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)
    results = [texts[i] for i in I[0]]
    context = "\n".join(results)

    prompt = f"Contexto:\n{context}\n\nPregunta: {query}\nRespuesta:"

    # Enviar a Ollama usando stream
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": True
        },
        stream=True
    )

    # Leer stream línea por línea y construir la respuesta completa
    full_response = ""
    for line in response.iter_lines():
        if line:
            try:
                json_line = line.decode("utf-8")
                data = json.loads(json_line)
                full_response += data.get("response", "")
            except Exception as e:
                print(f"[error leyendo línea]: {e}")

    return full_response
