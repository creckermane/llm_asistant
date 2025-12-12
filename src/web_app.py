# src/web_app.py
from flask import Flask, render_template, request, jsonify
import os
import logging

# Настройка логирования для всего Flask-приложения
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Импорты всех необходимых модулей для работы приложения
from src.data_loader import load_table_data
from src.qa_pipeline import ingest_data, ask_question, vector_store_instance

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "templates")

# Инициализация Flask-приложения
app = Flask(__name__, template_folder=TEMPLATES_DIR)


@app.route("/")
def chat():
    """
    Отображает главную страницу чата для пользователя.
    Соответствует требованию ТЗ: "Предоставить web-интерфейс (чат для пользователей)".
    """
    logging.info("Загрузка страницы чата.")
    return render_template("chat.html")


@app.route("/admin")
def admin():
    """
    Отображает административную панель для управления данными.
    Соответствует требованию ТЗ: "Предоставить web-интерфейс (админ панель для загрузки и диагностики данных)".
    """
    logging.info("Загрузка страницы админ-панели.")
    return render_template("admin.html")


@app.route("/api/ingest", methods=["POST"])
def api_ingest():
    """
    API-эндпоинт для загрузки и индексации данных.
    Читает данные из test_data.csv и передает их в qa_pipeline для инжеста.
    Соответствует требованию ТЗ: "Загрузка и индексация данных (ингест)".
    """
    logging.info("Получен запрос на инжест данных.")
    data_path = os.path.join(PROJECT_ROOT, "data", "test_data.csv")

    # Проверяем наличие файла данных
    if not os.path.exists(data_path):
        logging.error(f"Файл данных не найден: {data_path}")
        return jsonify({"error": f"Файл не найден: {data_path}"}), 400

    try:
        rows = load_table_data(data_path) # Загрузка данных
        ingest_data(rows) # Индексация данных
        logging.info(f"Успешно инжестировано {len(rows)} строк.")
        return jsonify({"status": "success", "rows": len(rows)})
    except Exception as e:
        logging.exception(f"Ошибка при инжесте данных: {e}") # Логируем исключение с traceback
        return jsonify({"error": str(e)}), 500


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """
    API-эндпоинт для обработки вопросов пользователя.
    Принимает текстовый вопрос, передает его в qa_pipeline и возвращает ответ LLM.
    Соответствует требованию ТЗ: "Принимать от пользователя текстовый вопрос в web-интерфейсе (чат)".
    """
    logging.info("Получен запрос на вопрос.")
    question = request.json.get("question")
    if not question:
        logging.warning("Получен пустой вопрос.")
        return jsonify({"error": "Нет вопроса"}), 400

    try:
        result = ask_question(question) # Обработка вопроса через RAG-пайплайн
        logging.info(f"Вопрос: '{question[:50]}...', Ответ LLM: '{result.get('answer', '')[:50]}...'")
        return jsonify(result)
    except Exception as e:
        logging.exception(f"Ошибка при обработке вопроса '{question[:50]}...': {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/reset_index", methods=["POST"])
def api_reset_index():
    """
    API-эндпоинт для очистки векторного индекса.
    Соответствует требованию ТЗ: "Поддерживать повторную индексацию (удаление)".
    """
    logging.info("Получен запрос на очистку индекса.")
    try:
        vector_store_instance.reset_collection()
        logging.info("Индекс успешно очищен.")
        return jsonify({"status": "success", "message": "Индекс успешно очищен."})
    except Exception as e:
        logging.exception(f"Ошибка при очистке индекса: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/index_stats", methods=["GET"])
def api_index_stats():
    """
    API-эндпоинт для получения статистики по векторному индексу.
    Соответствует требованию ТЗ: "Просмотр статистики индекса (кол-во чанков, размер и т.п.)".
    """
    logging.info("Получен запрос на статистику индекса.")
    try:
        stats = vector_store_instance.get_stats()
        logging.info(f"Статистика индекса: {stats}")
        return jsonify(stats)
    except Exception as e:
        logging.exception(f"Ошибка при получении статистики индекса: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logging.info("Запуск Flask-приложения...")
    # Запуск Flask-сервера.
    app.run(host="0.0.0.0", port=5000, debug=True)