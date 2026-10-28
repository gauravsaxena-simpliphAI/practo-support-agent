"""ChromaDB indexing and retrieval for Practo RAG Core.

Indexes both fixed-size-with-overlap and sentence-based chunks into separate
ChromaDB collections using local SentenceTransformers embeddings ('all-MiniLM-L6-v2').
"""

import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple

from knowledge_base.kb_docs import DOCUMENTS, get_kb_documents
from rag.chunking import fixed_size_chunking, sentence_based_chunking

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# Singleton embedding model
_EMBEDDING_MODEL = None

def get_embedding_model() -> SentenceTransformer:
    """Load and cache the local SentenceTransformer model."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDING_MODEL


def get_chroma_client() -> chromadb.ClientAPI:
    """Initialize persistent ChromaDB client."""
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR)


def build_and_index_collections() -> Tuple[chromadb.Collection, chromadb.Collection]:
    """Index both chunking strategies into separate ChromaDB collections."""
    client = get_chroma_client()
    model = get_embedding_model()
    docs = get_kb_documents()

    # Strategy 1: Fixed-size with overlap
    fixed_chunks = fixed_size_chunking(docs, chunk_size=220, overlap=40)
    fixed_texts = [c["text"] for c in fixed_chunks]
    fixed_embeddings = model.encode(fixed_texts, show_progress_bar=False).tolist()

    fixed_collection = client.get_or_create_collection(
        name="practo_kb_fixed",
        metadata={"hnsw:space": "cosine"}
    )
    fixed_collection.upsert(
        ids=[c["chunk_id"] for c in fixed_chunks],
        embeddings=fixed_embeddings,
        documents=fixed_texts,
        metadatas=[{
            "doc_id": c["doc_id"],
            "topic": c["topic"],
            "title": c["title"],
            "chunk_idx": c["chunk_idx"],
            "strategy": c["strategy"],
        } for c in fixed_chunks]
    )

    # Strategy 2: Sentence-based
    sentence_chunks = sentence_based_chunking(docs, sentences_per_chunk=1)
    sentence_texts = [c["text"] for c in sentence_chunks]
    sentence_embeddings = model.encode(sentence_texts, show_progress_bar=False).tolist()

    sentence_collection = client.get_or_create_collection(
        name="practo_kb_sentence",
        metadata={"hnsw:space": "cosine"}
    )
    sentence_collection.upsert(
        ids=[c["chunk_id"] for c in sentence_chunks],
        embeddings=sentence_embeddings,
        documents=sentence_texts,
        metadatas=[{
            "doc_id": c["doc_id"],
            "topic": c["topic"],
            "title": c["title"],
            "chunk_idx": c["chunk_idx"],
            "strategy": c["strategy"],
        } for c in sentence_chunks]
    )

    print(f"Indexed {len(fixed_chunks)} fixed chunks into 'practo_kb_fixed'.")
    print(f"Indexed {len(sentence_chunks)} sentence chunks into 'practo_kb_sentence'.")
    return fixed_collection, sentence_collection


def retrieve_chunks(
    query: str,
    collection_name: str = "practo_kb_sentence",
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Retrieve top-k chunks from specified collection with cosine similarity.

    Args:
        query: User input query.
        collection_name: 'practo_kb_fixed' or 'practo_kb_sentence'.
        top_k: Number of chunks to retrieve.

    Returns:
        List of dicts: {chunk_id, text, metadata, distance, similarity}
    """
    client = get_chroma_client()
    model = get_embedding_model()
    collection = client.get_collection(name=collection_name)

    query_embedding = model.encode([query], show_progress_bar=False).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []
    if results and results["ids"] and len(results["ids"][0]) > 0:
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            # Chroma with cosine space returns cosine distance = 1 - cosine_similarity
            similarity = max(0.0, min(1.0, 1.0 - distance))
            retrieved.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": float(distance),
                "similarity": float(similarity),
            })

    return retrieved


if __name__ == "__main__":
    fixed_coll, sent_coll = build_and_index_collections()
    sample_query = "What is the cancellation window for appointments?"
    res_fixed = retrieve_chunks(sample_query, "practo_kb_fixed", top_k=2)
    res_sent = retrieve_chunks(sample_query, "practo_kb_sentence", top_k=2)
    print("\nSample Query:", sample_query)
    print("Fixed Collection Top Result:", res_fixed[0]["similarity"], res_fixed[0]["text"][:80])
    print("Sentence Collection Top Result:", res_sent[0]["similarity"], res_sent[0]["text"][:80])
