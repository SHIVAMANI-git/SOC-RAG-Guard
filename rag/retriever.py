import json
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent
KNOWLEDGE_FILE = ROOT / "knowledge_base" / "cybersecurity_evidence.json"
CACHE_FILE = ROOT / "knowledge_base" / "embeddings_cache.json"

OLLAMA_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "embeddinggemma"


def load_documents():
    with KNOWLEDGE_FILE.open("r", encoding="utf-8-sig") as f:
        documents = json.load(f)

    if not documents:
        raise ValueError("The evidence corpus is empty.")

    required = {"source_id", "title", "source_url", "text"}
    for doc in documents:
        missing = required - set(doc)
        if missing:
            raise ValueError(
                f"Document {doc.get('source_id', '<unknown>')} "
                f"is missing fields: {sorted(missing)}"
            )

    return documents


def embed_texts(texts):
    """Generate embeddings through the local Ollama API."""
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=180,
    )
    response.raise_for_status()

    embeddings = response.json().get("embeddings")
    if not embeddings or len(embeddings) != len(texts):
        raise RuntimeError("Ollama returned an unexpected number of embeddings.")

    vectors = np.asarray(embeddings, dtype=np.float32)

    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise RuntimeError("Invalid embedding vectors returned by Ollama.")

    return vectors


def get_document_embeddings(documents):
    """Reuse the cache only when its source IDs and text still match."""
    if CACHE_FILE.exists():
        try:
            cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            same_docs = (
                cache.get("model") == EMBED_MODEL
                and cache.get("source_ids") == [d["source_id"] for d in documents]
                and cache.get("texts") == [d["text"] for d in documents]
            )

            if same_docs:
                vectors = np.asarray(cache["embeddings"], dtype=np.float32)
                if (
                    vectors.ndim == 2
                    and vectors.shape[0] == len(documents)
                    and np.isfinite(vectors).all()
                ):
                    print("Loaded cached document embeddings.")
                    return vectors
        except (ValueError, KeyError, TypeError):
            pass

    print(f"Embedding {len(documents)} evidence documents...")
    vectors = embed_texts([d["text"] for d in documents])

    cache_data = {
        "model": EMBED_MODEL,
        "source_ids": [d["source_id"] for d in documents],
        "texts": [d["text"] for d in documents],
        "embeddings": vectors.tolist(),
    }

    CACHE_FILE.write_text(
        json.dumps(cache_data),
        encoding="utf-8",
    )
    print(f"Embedding cache saved: {CACHE_FILE}")

    return vectors


def retrieve(query, top_k=3):
    """Return the top-k evidence documents ranked by cosine similarity."""
    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    documents = load_documents()
    document_vectors = get_document_embeddings(documents)
    query_vector = embed_texts([query])[0]

    if document_vectors.shape[1] != query_vector.shape[0]:
        raise RuntimeError("Query and document embedding dimensions differ.")

    document_norms = np.linalg.norm(document_vectors, axis=1)
    query_norm = np.linalg.norm(query_vector)

    if query_norm == 0 or np.any(document_norms == 0):
        raise RuntimeError("A zero-length embedding was returned.")

    scores = (document_vectors @ query_vector) / (document_norms * query_norm)

    order = np.argsort(scores)[::-1][: min(top_k, len(documents))]
    results = []

    for index in order:
        doc = documents[int(index)]
        results.append(
            {
                "source_id": doc["source_id"],
                "title": doc["title"],
                "source_url": doc["source_url"],
                "category": doc.get("category", "unspecified"),
                "similarity": float(scores[index]),
                "text": doc["text"],
            }
        )

    return results


def main():
    query = (
        "A network detector flags unusual connections. "
        "What evidence should an analyst validate and preserve "
        "before declaring a confirmed security incident?"
    )

    print("\nSOC query:")
    print(query)

    results = retrieve(query, top_k=3)

    print("\nRetrieved evidence:")
    for rank, item in enumerate(results, start=1):
        print(f"\n[{rank}] {item['source_id']}")
        print(f"Title: {item['title']}")
        print(f"Similarity: {item['similarity']:.4f}")
        print(f"Source: {item['source_url']}")
        print(f"Evidence: {item['text']}")


if __name__ == "__main__":
    main()
