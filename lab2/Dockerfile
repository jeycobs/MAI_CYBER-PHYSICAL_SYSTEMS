# взял обычный debian-базовый образ питона
FROM python:3.12-bullseye

WORKDIR /app

#ставим curl чтобы скачать олламу
RUN apt-get update && apt-get install -y curl zstd
RUN curl -fsSL https://ollama.com/install.sh | sh

#сначала зависимости
COPY requirements.txt .
RUN pip install -r requirements.txt

#потом весь код
COPY . .

#даем права на запуск
RUN chmod +x start.sh

#стартуем
CMD ["./start.sh"]
