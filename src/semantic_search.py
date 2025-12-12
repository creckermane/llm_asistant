# src/semantic_search.py
import logging
from typing import List, Dict, Any
from src.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def retrieve_context(query: str, vector_store: VectorStore, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Извлекает релевантные чанки из векторного хранилища на основе запроса.
    """
    logging.info(f"Запуск семантического поиска для запроса: '{query[:50]}...' (top_k={top_k})")
    try:
        retrieved_chunks = vector_store.search(query, top_k=top_k)

        # Добавляем ID к каждому чанку для дедупликации
        for i, chunk in enumerate(retrieved_chunks):
            chunk["id"] = f"{chunk['metadata'].get('row_id')}_{i}_{hash(chunk['text'])}"  # Уникальный ID для чанка

        return retrieved_chunks
    except Exception as e:
        logging.error(f"❌ Ошибка при выполнении семантического поиска: {e}")
        raise