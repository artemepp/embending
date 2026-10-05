import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# Лучшая модель из исследования
MODEL_ID = "paraphrase-multilingual-mpnet-base-v2"
CACHE_FILE = Path("embeddings_cache") / "corpus_MPNet-Multilingual.npy"

app = FastAPI(title="Semantic code search")

# Загрузка данных
corpus = json.load(open("data/code_corpus.json", encoding="utf-8"))
documents = [f"{f['function_name']}: {f['description']}\n{f['code']}" for f in corpus]

# Загрузка модели
model = SentenceTransformer(MODEL_ID)

# Эмбеддинги корпуса: берём из кэша ноутбука, если он подходит
if CACHE_FILE.exists():
    corpus_emb = np.load(CACHE_FILE)
else:
    corpus_emb = None

if corpus_emb is None or len(corpus_emb) != len(documents):
    corpus_emb = normalize(model.encode(documents, batch_size=32))
    CACHE_FILE.parent.mkdir(exist_ok=True)
    np.save(CACHE_FILE, corpus_emb)


@app.get("/search")
def search(q: str = Query(..., min_length=1), k: int = 3):
    query_emb = normalize(model.encode([q]))[0]
    scores = corpus_emb @ query_emb  # косинусное сходство (векторы нормализованы)
    top_idx = np.argsort(-scores)[:k]

    return {
        "query": q,
        "results": [
            {
                "rank": rank,
                "id": corpus[i]["id"],
                "function_name": corpus[i]["function_name"],
                "language": corpus[i]["language"],
                "category": corpus[i]["category"],
                "description": corpus[i]["description"],
                "score": round(float(scores[i]), 4),
            }
            for rank, i in enumerate(top_idx, start=1)
        ],
    }