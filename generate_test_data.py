# generate_test_data.py
import pandas as pd
import numpy as np
import random
import os


def generate_test_data(num_rows=100, output_file="data/test_data.csv"):
    """
    Генерирует синтетические табличные данные для тестирования.
    """
    data = []

    # Списки возможных значений для полей
    periods = [f"p{i}" for i in range(1, 7)]
    customers = [
        f"c{random.randint(2000, 3000)} Склад ТД [{random.choice(['ПФО', 'УФО', 'ЦФО', 'ЮФО'])}:{random.choice(['Уфа', 'Екатеринбург', 'Москва', 'Казань', 'Ростов', 'Краснодар'])}]"
        for _ in range(20)]
    products = [f"i{random.randint(6000000, 7000000)} Арматура {chr(65 + i)}" for i in range(10)] + \
               [f"i{random.randint(6000000, 7000000)} Профиль {chr(65 + i)}" for i in range(5)]

    for i in range(num_rows):
        # Генерация одной строки данных
        row = {
            "id": i + 1,
            "Период_планирования": random.choice(periods),
            "Покупатель_спроса": random.choice(customers),
            "Продукт_спроса": random.choice(products),
            "Минимальный_заказ": random.randint(10, 100),
            "Максимальный_заказ": random.randint(100, 500),
            "Фактически_удовлетворённый_объём": random.randint(50, 400),
            "Процент_удовлетворения_спроса": round(random.uniform(0.7, 1.0), 2),
            "Выручка_за_единицу": round(random.uniform(100, 1000), 2),
            "Общая_выручка_по_заказу": round(random.uniform(5000, 50000), 2),
            "Штрафы_за_недопоставку": round(random.uniform(0, 500), 2) if random.random() < 0.3 else 0.0,
            "Штрафы_за_перепоставку": round(random.uniform(0, 200), 2) if random.random() < 0.1 else 0.0,
            "Штрафы_на_партию": round(random.uniform(0, 100), 2) if random.random() < 0.05 else 0.0,
        }
        data.append(row)

    # Создание DataFrame и сохранение в CSV
    df = pd.DataFrame(data)

    # Создаем папку data, если ее нет
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"✅ Сгенерировано {num_rows} строк данных и сохранено в {output_file}")


if __name__ == "__main__":
    generate_test_data()
