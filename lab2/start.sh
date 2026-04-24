#!/bin/bash

#запускаем сервер олламы в фоне
ollama serve &

#ждем пока он поднимется
sleep 5

#качаем модель
ollama pull qwen2.5:0.5b

uvicorn app:app --host 0.0.0.0 --port 8000
