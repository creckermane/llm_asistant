# run.py
import os
import logging
from src.web_app import app

# Настройка логирования для запускающего скрипта
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    logging.info("Запуск Flask-приложения через run.py...")
    app.run(host="0.0.0.0", port=5000, debug=True)
