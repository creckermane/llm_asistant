# # src/config.py

#  Настройки для RAG-пайплайна
RETRIEVAL_TOP_K = 15 # Извлекаемые чанки

# Настройки для мульти-запросного поиска
ENABLE_MULTI_QUERY_RETRIEVAL = True # Включаем мульти-запросный поиск
MULTI_QUERY_GENERATION_COUNT = 3 # Сколько альтернативных запросов генерировать


# Векторизация
EMBEDDING_MODEL_NAME = "nomic-embed-text:latest"   # ← именно так, как в Ollama
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# LLM
OLLAMA_MODEL = "gemma3:4b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Векторное хранилище
VECTOR_DB_PATH = "./chroma_db"
COLLECTION_NAME = "demand_data_collection"
