# src/data_loader.py
import os
import pandas as pd
from typing import List, Dict
import logging

# Настройка логирования для модуля
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_table_data(file_path: str) -> List[Dict]:
    """
    Загружает данные из CSV-файла и возвращает их в виде списка словарей.
    Каждый словарь представляет одну строку таблицы.
    Соответствует требованию ТЗ: "Принимать от администратора данные для индексации".
    """
    logging.info(f"Попытка загрузки данных из {file_path}")
    try:
        # Чтение CSV-файла с использованием pandas
        df = pd.read_csv(file_path)
        # Преобразование DataFrame в список словарей, где каждый словарь - это строка
        data = df.to_dict(orient='records')
        logging.info(f"✅ Успешно загружено {len(data)} строк из {file_path}")
        return data
    except FileNotFoundError:
        logging.error(f"❌ Ошибка: Файл не найден по пути {file_path}")
        raise # Перевыбрасываем исключение для обработки выше
    except Exception as e:
        logging.exception(f"❌ Ошибка при загрузке данных из CSV: {e}")
        raise # Перевыбрасываем исключение для обработки выше

# Пример использования модуля (для автономного тестирования)
if __name__ == "__main__":
    try:
        # Для корректного пути при запуске из src/
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        test_data_path = os.path.join(project_root, "data", "test_data.csv")

        if not os.path.exists(test_data_path):
            print(f"Файл {test_data_path} не найден. Пожалуйста, сгенерируйте его с помощью generate_test_data.py.")
        else:
            test_rows = load_table_data(test_data_path)
            if test_rows:
                print("\nПервая строка загруженных данных:")
                print(test_rows[0])
            else:
                print("\nЗагруженные данные пусты.")
    except Exception as e:
        print(f"Не удалось загрузить тестовые данные: {e}")
