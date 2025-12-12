# src/qa_pipeline.py
import logging
from typing import List, Dict, Any
from src.vector_store import VectorStore
from src.llm_interface import OllamaLLM
from src.text_formatter import format_row_as_text, chunk_text
from src.semantic_search import retrieve_context
from src.config import (
    OLLAMA_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
    RETRIEVAL_TOP_K, ENABLE_MULTI_QUERY_RETRIEVAL, MULTI_QUERY_GENERATION_COUNT
)
import re

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Инициализация компонентов
vector_store_instance = VectorStore()
llm_instance = OllamaLLM(model_name=OLLAMA_MODEL)

# Системный промпт для LLM
SYSTEM_PROMPT = """Ты — полезный ассистент, который отвечает на вопросы по табличным данным о спросе.
Твоя задача — извлекать точную и полную информацию из предоставленного контекста.

Правила:
1.  **Используй ТОЛЬКО предоставленный контекст.** Не используй свои общие знания.
2.  **Если ответ НЕ содержится в контексте,** так и скажи: "Извините, я не могу ответить на этот вопрос на основе предоставленных данных."
3.  **Отвечай максимально полно и точно.** Если в контексте есть несколько значений для одного запроса, перечисли их все или укажи диапазон, если это уместно.
4.  **При ответе на вопросы, связанные с числовыми значениями** (выручка, штрафы, объемы, процент удовлетворения), старайся явно указывать эти значения из контекста, включая единицы измерения или тип (например, "Общая выручка по заказу составляет 12345.67").
5.  **Если вопрос требует АГРЕГАЦИИ или ВЫЧИСЛЕНИЙ** (например, "средний", "сумма", "максимальный"), и ты не видишь готового агрегированного значения в контексте, отвечай: "Извините, я не могу выполнить вычисления или агрегацию данных. Я могу только извлекать информацию, которая явно присутствует в предоставленном контексте. Для получения среднего значения по Арматура J, я нашел следующие проценты удовлетворения: [перечисли найденные значения]."
6.  **Если вопрос не является запросом информации из контекста** (например, "привет", "как дела?"), отвечай: "Я могу отвечать только на вопросы, связанные с данными о спросе, предоставленными мне."

Контекст:
{context}

Вопрос: {question}
"""

# Промпт для генерации альтернативных запросов (для мульти-запросного поиска)
MULTI_QUERY_PROMPT = """Ты — эксперт по генерации поисковых запросов. Твоя задача — сгенерировать {count} различных, но семантически похожих поисковых запросов на основе исходного пользовательского запроса. Эти запросы будут использоваться для поиска релевантной информации в векторной базе данных.
Сгенерируй запросы, которые могут раскрыть разные аспекты исходного запроса или использовать синонимы.
Выводи каждый запрос на новой строке. Не добавляй никаких других слов, кроме самих запросов.

Пример:
Пользовательский запрос: Какова выручка для продукта Арматура J?
Сгенерированные запросы:
Выручка Арматура J
Сколько заработали на Арматура J?
Финансовые показатели Арматура J

Пользовательский запрос: {original_query}
Сгенерированные запросы:
"""


def ingest_data(rows: List[Dict]):
    """
    Обрабатывает и индексирует данные в векторном хранилище.
    """
    logging.info(f"Начинаю инжест {len(rows)} строк данных...")
    vector_store_instance.reset_collection()  # Очищаем коллекцию перед новым инжестом

    all_chunks = []
    all_metadatas = []
    all_ids = []

    for i, row in enumerate(rows):
        # Форматируем строку в текст
        text_data = format_row_as_text(row)

        # Разбиваем текст на чанки
        chunks = chunk_text(text_data, CHUNK_SIZE, CHUNK_OVERLAP)

        for j, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            # Добавляем метаданные, включая row_id для отслеживания источника
            all_metadatas.append({"row_id": row.get("row_id", i + 1), "source_file": "test_data.csv"})
            all_ids.append(f"doc_{row.get('row_id', i + 1)}_chunk_{j}")

    logging.info(f"Добавляю {len(all_chunks)} чанков в векторное хранилище...")
    vector_store_instance.add_chunks(all_chunks, all_metadatas, all_ids)
    logging.info("✅ Данные успешно добавлены в векторное хранилище.")


def _generate_alternative_queries(original_query: str, count: int) -> List[str]:
    """
    Генерирует альтернативные поисковые запросы с помощью LLM.
    """
    logging.info(f"Генерация {count} альтернативных запросов для: '{original_query}'")
    prompt = MULTI_QUERY_PROMPT.format(count=count, original_query=original_query)
    try:
        response = llm_instance.generate(prompt, temperature=0.4)  # Используем температуру для разнообразия
        queries = [q.strip() for q in response.split('\n') if q.strip()]

        if original_query not in queries:
            queries.insert(0, original_query)
        logging.info(f"Сгенерированные запросы: {queries}")
        return queries
    except Exception as e:
        logging.error(f"Ошибка при генерации альтернативных запросов: {e}")
        return [original_query]


def ask_question(question: str) -> Dict[str, Any]:
    """
    Обрабатывает вопрос пользователя, выполняет RAG-пайплайн и возвращает ответ.
    """
    logging.info(f"Начинаю обработку вопроса: '{question}'")

    all_retrieved_chunks = []
    queries_to_search = [question]

    if ENABLE_MULTI_QUERY_RETRIEVAL:
        alternative_queries = _generate_alternative_queries(question, MULTI_QUERY_GENERATION_COUNT)
        queries_to_search.extend(alternative_queries)
        # Удаляем мусор и сохраняем порядок
        queries_to_search = list(dict.fromkeys(queries_to_search))

    for q in queries_to_search:
        logging.info(f"Запуск семантического поиска для запроса: '{q}' (top_k={RETRIEVAL_TOP_K})")
        try:
            retrieved_chunks_for_query = retrieve_context(q, vector_store_instance, top_k=RETRIEVAL_TOP_K)
            all_retrieved_chunks.extend(retrieved_chunks_for_query)
        except Exception as e:
            logging.error(f"❌ Ошибка при выполнении семантического поиска для запроса '{q}': {e}")

    # Удаляем дубликаты чанков (если один и тот же чанк найден по разным запросам)
    # Используем ID чанка для уникальности
    unique_chunks_map = {chunk["id"]: chunk for chunk in all_retrieved_chunks if "id" in chunk}
    final_retrieved_chunks = list(unique_chunks_map.values())

    if not final_retrieved_chunks:
        logging.warning("Не найдено релевантных чанков для вопроса.")
        return {
            "answer": "Извините, я не могу ответить на этот вопрос на основе предоставленных данных, так как не найдено релевантной информации.",
            "sources": []}

    # Сортируем чанки по релевантности (по расстоянию)
    final_retrieved_chunks.sort(key=lambda x: x.get("distance", 0.0))

    # Формируем контекст для LLM
    context_texts = [chunk["text"] for chunk in final_retrieved_chunks]
    context = "\n\n".join(context_texts)
    logging.info(f"Найден контекст (первые 200 символов): {context[:200]}...")

    # Формируем промпт для LLM
    prompt = SYSTEM_PROMPT.format(context=context, question=question)

    # Получаем ответ от LLM
    try:
        llm_answer = llm_instance.generate(prompt)
        logging.info(f"Ответ LLM: {llm_answer[:200]}...")
    except Exception as e:
        logging.error(f"❌ Ошибка при получении ответа от LLM: {e}")
        llm_answer = "Извините, произошла ошибка при генерации ответа."

    # Извлекаем источники
    sources = []
    for chunk in final_retrieved_chunks:
        row_id = chunk["metadata"].get("row_id", "N/A")
        source_file = chunk["metadata"].get("source_file", "N/A")
        sources.append(f"Строка {row_id} из {source_file}")

    # Уникальные и отсортированные источники
    unique_sources = sorted(list(set(sources)))

    # Пост-обработка ответа для агрегации (если LLM не справилась)
    # Этот блок кода будет пытаться извлечь числовые значения из контекста
    # и выполнить агрегацию, если вопрос явно об этом.
    # Это "хак", так как LLM сама не умеет считать.
    if "средний процент удовлетворения спроса" in question.lower() and "арматура" in question.lower():
        product_names = re.findall(r"арматура [A-Za-z]", question, re.IGNORECASE)
        if product_names:
            product_values = {}
            for product_name in product_names:
                product_name_clean = product_name.strip()

                pattern = rf"{re.escape(product_name_clean)};\s*Процент удовлетворения спроса\s*–\s*(\d+\.?\d*)"

                found_percentages = []
                for chunk_text in context_texts:
                    matches = re.findall(pattern, chunk_text, re.IGNORECASE)
                    for match in matches:
                        try:
                            found_percentages.append(float(match))
                        except ValueError:
                            pass  # Игнорируем нечисловые значения

                if found_percentages:
                    avg_percentage = sum(found_percentages) / len(found_percentages)
                    product_values[product_name_clean] = avg_percentage
                else:
                    product_values[product_name_clean] = None

            if product_values:
                final_agg_answer = []
                for prod, avg_val in product_values.items():
                    if avg_val is not None:
                        final_agg_answer.append(
                            f"Средний процент удовлетворения спроса для продукта {prod} составляет {avg_val:.2f}.")
                    else:
                        final_agg_answer.append(
                            f"Не удалось найти процент удовлетворения спроса для продукта {prod} в контексте.")

                if final_agg_answer:
                    llm_answer = " ".join(final_agg_answer) + "\n\n" + llm_answer  # Добавляем к ответу LLM
                    logging.info(f"Пост-обработка добавила агрегированный ответ: {llm_answer[:200]}...")

    return {"answer": llm_answer, "sources": unique_sources}