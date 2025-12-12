# src/llm_interface.py
import requests
import logging
from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL

# Настройка логирования для модуля
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class OllamaLLM:
    """
    Класс для взаимодействия с локальной моделью Ollama.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.base_url = OLLAMA_BASE_URL
        logging.info(f"Инициализация OllamaLLM с моделью '{self.model_name}' по URL '{self.base_url}'")

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Отправляет промпт в локальную модель Ollama и возвращает ответ.
        """
        return self._call_ollama(prompt, temperature)

    def _call_ollama(self, prompt: str, temperature: float) -> str:
        """
        Выполняет запрос к локальному Ollama API для получения ответа от LLM.
        """
        try:
            logging.info(f"📨 Отправляю в Ollama (модель: {self.model_name}, temp: {temperature}): {prompt[:100]}...")

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature
                    }
                },
                timeout=180
            )

            # Проверяем HTTP-статус ответа
            if response.status_code != 200:
                logging.error(f"❌ Ollama вернул статус {response.status_code}: {response.text}")
                return "[Ошибка: Не удалось получить ответ от модели Ollama]"

            data = response.json()

            if "response" not in data:
                logging.warning(f"⚠️ Нет поля 'response' в ответе Ollama: {data}")
                return "[Ошибка: некорректный ответ от LLM Ollama]"

            raw_text = data["response"].strip()

            # Удаляем возможные шум/мусор от модели
            clean_text = raw_text.split("</")[0].strip() if "</" in raw_text else raw_text
            clean_text = clean_text.replace("Assistant:", "").strip()

            logging.info(f"✅ Получен ответ от Ollama: {clean_text[:100]}...")
            return clean_text

        except requests.exceptions.ConnectionError:
            logging.error(
                "🔴 Ошибка подключения: не могу подключиться к Ollama. Убедитесь, что ollama запущен ('ollama serve').")
            return "[Ошибка: не удается подключиться к Ollama. Запустите 'ollama serve'?]"
        except requests.exceptions.Timeout:
            logging.error(
                "⏰ Таймаут (180с) при обращении к Ollama. Модель слишком медленная или запрос слишком большой.")
            return "[Ошибка: таймаут ответа от модели Ollama]"
        except Exception as e:
            logging.exception(f"🔴 Неожиданная ошибка при вызове Ollama: {e}")
            return f"[Внутренняя ошибка Ollama: {str(e)}]"


# Пример использования модуля (для автономного тестирования)
if __name__ == "__main__":
    print("--- Тестирование LLM Interface (Ollama) ---")


    llm_test_instance = OllamaLLM(model_name="gemma3:1b")  # Используйте вашу модель

    test_prompt_simple = "Привет, как дела? Ответь кратко."
    print(f"\nВопрос: {test_prompt_simple}")
    response_simple = llm_test_instance.generate(test_prompt_simple)
    print(f"Ответ LLM: {response_simple}")

    test_prompt_complex = "Расскажи о себе, как о языковой модели."
    print(f"\nВопрос: {test_prompt_complex}")
    response_complex = llm_test_instance.generate(test_prompt_complex)
    print(f"Ответ LLM: {response_complex}")
