# src/text_formatter.py
import tiktoken
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def format_row_as_text(row: Dict) -> str:
    """
    Форматирует одну строку данных в текстовое представление для индексации.
    """
    formatted_parts = []
    # Исключаем 'row_id' из форматирования, так как это внутренний идентификатор
    for key, value in row.items():
        if key == "row_id":
            continue
        # Делаем более читабельные названия полей, если они есть
        if key == "Период планирования":
            formatted_parts.append(f"Период планирования – {value}")
        elif key == "Покупатель спроса":
            formatted_parts.append(f"Покупатель спроса – {value}")
        elif key == "Продукт спроса":
            formatted_parts.append(f"Продукт спроса – {value}")
        elif key == "Общая выручка по заказу":
            formatted_parts.append(f"Общая выручка по заказу – {value}")
        elif key == "Штрафы за недопоставку":
            formatted_parts.append(f"Штрафы за недопоставку – {value}")
        elif key == "Процент удовлетворения спроса":
            formatted_parts.append(f"Процент удовлетворения спроса – {value}")
        else:
            formatted_parts.append(f"{key} – {value}")

    return "; ".join(formatted_parts) + "."


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Разбивает текст на чанки с заданным размером и перекрытием.
    Использует tiktoken для подсчета токенов.
    """

    try:
        encoding = tiktoken.get_encoding("cl100k_base")  # Универсальный кодировщик
    except Exception:
        logging.warning("Не удалось загрузить tiktoken кодировщик, используя простой split по словам.")
        words = text.split()
        if not words:
            return []

        chunks = []
        current_chunk_words = []
        current_chunk_len = 0

        for word in words:
            word_len = len(word.split())  # Приближенно, 1 слово = 1 токен
            if current_chunk_len + word_len <= chunk_size:
                current_chunk_words.append(word)
                current_chunk_len += word_len
            else:
                chunks.append(" ".join(current_chunk_words))
                # Добавляем перекрытие
                overlap_words = current_chunk_words[-int(
                    chunk_overlap / (chunk_size / len(current_chunk_words) + 0.001)):] if current_chunk_words else []
                current_chunk_words = overlap_words + [word]
                current_chunk_len = len(" ".join(current_chunk_words).split())
        if current_chunk_words:
            chunks.append(" ".join(current_chunk_words))
        return chunks

    tokens = encoding.encode(text)

    if not tokens:
        return []

    chunks = []
    for i in range(0, len(tokens), chunk_size - chunk_overlap):
        chunk_tokens = tokens[i: i + chunk_size]
        chunks.append(encoding.decode(chunk_tokens))

    return chunks