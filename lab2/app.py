from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

class LLMRequest(BaseModel):
    system_prompt: str
    text: str

@app.post("/predict")
def predict_spam(data: LLMRequest):
    """
    эндпоинт для предсказания спама.
    пересылает запрос в локальный сервер Ollama.
    возвращает ответ в формате JSON.
    """
    #стучимся в локальную олламу
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": "qwen2.5:0.5b",
        "system": data.system_prompt,
        "prompt": f"Text: {data.text}",
        "stream": False,
        "format": "json" # принудительно просим json
    }
    
    try:
        response = requests.post(url, json=payload)
        res_json = response.json()
        return {"response": res_json.get("response", "{}")}
    except Exception as e:
        # print("ошибка подключения:", e)
        return {"error": str(e)}
