# KekBootCamp

> Kek – потому что почему бы и нет? 🚀

## Установка и запуск

### 1. Клонируй репозиторий
```bash
git clone https://github.com/yourusername/research-pro-mode.git
cd research-pro-mode
```

### 2. Создай виртуальное окружение
```bash
python -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate
```

### 3. Установи зависимости
```bash
pip install streamlit langchain-core langchain-openai tavily-python requests beautifulsoup4 pydantic pandas python-dotenv
```

### 4. Настрой API-ключи
Создай файл `.env` в корне проекта и добавь:
```
API_KEY=твой_ключ_от_cloud_ru
TAVILY_API_KEY=твой_ключ_от_tavily
DEBUG_MODE=true  # Опционально, для отладки
```

### 5. Запусти приложение
```bash
streamlit run app.py
```

Открой браузер по адресу `http://localhost:8501`. Выбери режим, введи запрос и жми **🚀 Запустить поиск**!

> **Совет:** Если что-то сломается – проверь ключи в `.env` и Python 3.10+. Для бенчмарков кликни в сайдбаре. Удачи, kekster! 😎
