import os
import shutil
import base64
import uuid
import json
import requests
import pandas as pd
from typing import List, Optional, Dict, Any, Union

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Clients Open Source
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from pdf2image import convert_from_path

app = FastAPI(title="Local RAG API")

# config
OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME: str = "local_docs"

# On charge le modèle globalement
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
client_qdrant = QdrantClient(url=QDRANT_URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialisation DB
try:
    client_qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
    )
    print(f"Collection '{COLLECTION_NAME}' créée avec succès.")
except Exception as e:
    # On affiche l'info et on continue sans planter le programme.
    print(f"Info Qdrant : La collection existe probablement déjà. On continue. (Log: {e})")


class QueryRequest(BaseModel):
    question: str


# --- FONCTIONS TYPÉES ---

def get_local_embedding(text: str) -> List[float]:
    """Génère un vecteur d'embedding (liste de floats) pour un texte donné."""
    return embedding_model.encode(text).tolist()


def generate_ollama(
        model: str,
        prompt: str,
        image_base64: Optional[str] = None,
        format_json: bool = False
) -> str:
    """Appel générique à l'API Ollama. Retourne la chaîne générée."""
    url = f"{OLLAMA_URL}/api/generate"

    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    if image_base64:
        payload["images"] = [image_base64]
    if format_json:
        payload["format"] = "json"

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Lève une erreur si code HTTP != 200
        return response.json().get("response", "")
    except Exception as e:
        print(f"Erreur Ollama: {e}")
        return "Erreur lors de la génération par le modèle local."


def analyze_image_with_llava(image_path: str) -> str:
    """Utilise LLaVA pour décrire une image stockée sur disque."""
    with open(image_path, "rb") as img:
        b64_string = base64.b64encode(img.read()).decode('utf-8')

    prompt = (
        "Décris cette image de document en détail. "
        "Si c'est un tableau, liste les colonnes et quelques valeurs. "
        "Si c'est du texte, résume le contenu principal."
    )
    return generate_ollama("llava", prompt, image_base64=b64_string)


# --- ENDPOINTS ---

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Endpoint pour ingérer CSV ou PDF."""
    file_path = f"data/{file.filename}"

    # Écriture du fichier
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    filename = str(file.filename).lower()
    points: List[models.PointStruct] = []

    # 1. Traitement CSV
    if filename.endswith('.csv'):
        df = pd.read_csv(file_path)
        # On limite si le CSV est énorme (1000 lignes max pour la démo)
        for idx, row in df.head(1000).iterrows():
            # Conversion de la Série pandas en dictionnaire pour itération propre
            row_dict = row.to_dict()
            content = ", ".join([f"{col}: {val}" for col, val in row_dict.items()])

            vector = get_local_embedding(content)

            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"type": "data_row", "content": content, "source": filename}
            ))

    # 2. Traitement PDF / Images
    elif filename.endswith('.pdf'):
        images = convert_from_path(file_path)

        for i, img in enumerate(images):
            img_path = f"data/{file.filename}_p{i}.jpg"
            img.save(img_path, 'JPEG')

            # Vision Locale (LLaVA)
            description = analyze_image_with_llava(img_path)

            vector = get_local_embedding(description)

            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "type": "visual_doc",
                    "content": description,
                    "image_path": img_path,
                    "source": filename
                }
            ))

    # Upsert dans Qdrant
    if points:
        client_qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

    return {"status": "success", "indexed": len(points), "mode": "local_ollama"}


@app.post("/ask")
async def ask_agent(request: QueryRequest) -> Dict[str, Union[str, Dict[str, Any], None]]:
    """Endpoint RAG : Recherche + Génération."""

    # 1. Retrieval
    query_vector = get_local_embedding(request.question)

    search_result = client_qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=3
    )

    context_str = "\n".join([f"- {res.payload['content']}" for res in search_result])

    # 2. Generation
    prompt = f"""
    Tu es un assistant expert en analyse de documents.
    Voici des informations contextuelles extraites de documents (CSV, PDF, Images) :
    {context_str}

    Question de l'utilisateur : {request.question}

    Réponds en format JSON uniquement avec cette structure :
    {{
        "answer": "Ta réponse textuelle ici basée sur le contexte.",
        "chart": {{ "type": "bar", "labels": ["A", "B"], "values": [10, 20] }} 
    }}
    Si aucun graphique n'est pertinent, mets "chart": null.
    """

    # Appel à Ollama (Llama 3)
    response_str = generate_ollama("llama3.2", prompt, format_json=True)

    try:
        data = json.loads(response_str)
        return {
            "text": data.get("answer", "Pas de réponse textuelle trouvée."),
            "chart_data": data.get("chart")
        }
    except json.JSONDecodeError:
        return {"text": response_str, "chart_data": None}