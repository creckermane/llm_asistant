# src/vector_store.py

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from typing import List, Dict
from src.config import (
    VECTOR_DB_PATH,
    COLLECTION_NAME,
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL_NAME
)


class VectorStore:
    def __init__(self):
        self.embedding_fn = OllamaEmbeddingFunction(
            url=f"{OLLAMA_BASE_URL}/api/embeddings",
            model_name=EMBEDDING_MODEL_NAME
        )
        self.client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn
        )

    def reset_collection(self):
        """
        Удаляет и пересоздаёт коллекцию.
        """
        self.client.delete_collection(name=COLLECTION_NAME)
        self.collection = self.client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn
        )

    def get_stats(self) -> dict:
        """
        Возвращает статистику по коллекции.
        """
        count = self.collection.count()
        return {
            "count": count,
            "collection_name": COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "db_path": VECTOR_DB_PATH
        }

    def add_chunks(self, chunks: List[str], metadatas: List[Dict], ids: List[str]):
        clean_meta = [{k: str(v) for k, v in m.items()} for m in metadatas]
        self.collection.add(documents=chunks, metadatas=clean_meta, ids=ids)

    def search(self, query: str, top_k: int = 15) -> List[Dict]:
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        return [
            {
                "text": doc or "",
                "metadata": meta or {},
                "distance": dist or 1.0
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]