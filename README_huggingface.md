# HuggingFace Math Task Processor

Этот скрипт обрабатывает математические задачи из `test_private.csv` с помощью модели Qwen 2.5 7B через HuggingFace API.

## Установка

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Создайте файл `.env` на основе `.env.template`:
```bash
cp .env.template .env
```

3. Получите токен HuggingFace API:
   - Зайдите на https://huggingface.co/settings/tokens
   - Создайте новый токен с правами на чтение
   - Добавьте токен в файл `.env`:
```
HUGGINGFACE_API_TOKEN=your_actual_token_here
```

## Использование

Запустите скрипт:
```bash
python huggingface_processor.py
```

## Что делает скрипт

1. **Загружает задачи** из `test_private.csv`
2. **Читает системный промпт** из `llm_parser/prompts/system.txt`
3. **Отправляет каждую задачу** в Qwen 2.5 7B через HuggingFace API
4. **Сохраняет ответы** в формате JSON в папку `huggingface_outputs/`

## Структура выходных файлов

### Индивидуальные файлы задач
- `huggingface_outputs/task_000001.json` - результат для первой задачи
- `huggingface_outputs/task_000002.json` - результат для второй задачи
- и т.д.

### Сводные файлы
- `huggingface_outputs/all_results_YYYYMMDD_HHMMSS.json` - все результаты в одном файле
- `huggingface_outputs/report_YYYYMMDD_HHMMSS.json` - отчет о выполнении

## Формат результата для каждой задачи

```json
{
  "task_id": "task_000001",
  "task": "Текст задачи",
  "prompt": "Полный промпт с системным сообщением",
  "response": "Ответ модели",
  "parsed_json": {...},  // Если ответ содержит валидный JSON
  "latency": 2.34,
  "attempts": 1,
  "timestamp": "2025-09-22T12:30:45.123456",
  "status": "success"
}
```

## Обработка ошибок

Скрипт автоматически:
- **Повторяет запросы** при временных ошибках (до 3 попыток)
- **Ждет загрузки модели** если она не готова (HTTP 503)
- **Соблюдает лимиты** API (HTTP 429)
- **Логирует все операции** в `huggingface_processor.log`

## Настройки

В коде можно изменить:
- `max_new_tokens`: максимальная длина ответа (по умолчанию 2048)
- `temperature`: креативность модели (по умолчанию 0.1)
- `top_p`: фильтрация токенов (по умолчанию 0.9)
- Количество одновременных запросов (по умолчанию 2)

## Мониторинг

Скрипт выводит прогресс в консоль и записывает подробные логи в файл:
```
2025-09-22 12:30:45 - INFO - Loaded 344 tasks from test_private.csv
2025-09-22 12:30:46 - INFO - Processing task task_000001
2025-09-22 12:30:48 - INFO - Task task_000001 completed successfully in 2.34s
2025-09-22 12:30:50 - INFO - Completed 10/344 tasks
```

## Примечания

- Модель Qwen 2.5 7B может потребовать время на загрузку при первом запросе
- HuggingFace Inference API имеет лимиты на количество запросов
- Для больших объемов данных рекомендуется использовать HuggingFace Pro или собственную инфраструктуру