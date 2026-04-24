#задание на 4 
# прототип детектора смс-спама на базе llm
#(отчет в report.md)
## описание
проект представляет собой микросервис для фильтрации спама. внутри контейнера крутится модель qwen2.5:0.5b через сервер ollama, обернутая в fastapi.

## системные требования
* docker и docker-compose
* python 3.10+ (для запуска скрипта исследования)
* файл с датасетом spam.csv в корневой папке

## запуск сервера
1. поднять контейнер (перед этим лучше на всякий запустить докер):
   docker-compose up --build
2. дождаться надписи о запуске uvicorn на порту 8000. модель скачается автоматически при первом запуске.

## проверка работоспособности
отправить запрос через терминал:
curl -x post http://localhost:8000/predict \
-h "content-type: application/json" \
-d '{"system_prompt": "you are a spam filter", "text": "win a free prize now, click here!"}'

## запуск исследования
для оценки метрик (вне докера):
1. установить зависимости: pip install pandas scikit-learn requests tqdm
2. запустить: python3 research.py
результаты тестов по техникам zero-shot, cot, few-shot и cot_few-shot отобразятся в консоли.
