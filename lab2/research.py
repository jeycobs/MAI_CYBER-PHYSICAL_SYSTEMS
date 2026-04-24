import pandas as pd
import requests
import json
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

def load_data(path: str, n_samples: int = 150):
    df = pd.read_csv(path, encoding='latin-1')
    df = df[['v1', 'v2']] # в этом датасете колонки называются v1 и v2
    df.columns = ['label', 'text']
    
    df['target'] = df['label'].apply(lambda x: 1 if x == 'spam' else 0)
    
    # берем кусочек, чтобы не ждать час
    df_sample = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
    return df_sample

def get_prompts():
    base = (
        "ты — система обнаружения СМС-спама. Твоя задача проанализировать текст на английском языке. "
        "выведи ТОЛЬКО валидный JSON со строго двумя ключами: "
        "'reasoning' (строка) и 'verdict' (число 0 или 1). "
        "0 — нормальное сообщение, 1 — спам (реклама, мошенничество)."
    )
    
    prompts = {
        "zero-shot": base,
        
        "cot": base + "\nсначала напиши свои логические рассуждения в поле 'reasoning' (проверь ссылки, срочность, деньги), затем укажи 'verdict'.",
        
        "few-shot": base + (
            "\nпримеры:\n"
            "текст: 'Hello, how are you?' -> {\"reasoning\": \"обычное личное сообщение, без подозрительных элементов.\", \"verdict\": 0}\n"
            "текст: 'FREE MONEY click here!' -> {\"reasoning\": \"обещание бесплатных денег и подозрительная ссылка.\", \"verdict\": 1}"
        ),
        
        "cot_few-shot": base + (
            "\nрассуждай логически шаг за шагом в поле 'reasoning' перед вынесением вердикта.\n"
            "примеры:\n"
            "Текст: 'Call me when you arrive.'\n"
            "Ответ: {\"reasoning\": \"простая личная просьба связаться по прибытии, нет ссылок и рекламы.\", \"verdict\": 0}\n"
            "Текст: 'URGENT! your account is blocked. Call 1-800-SPAM.'\n"
            "Ответ: {\"reasoning\": \"сообщение создает ложное чувство срочности (URGENT!) и просит позвонить на неизвестный номер для разблокировки аккаунта. Это классический фишинг.\", \"verdict\": 1}"
        )
    }
    return prompts

def get_verdict_from_llm(text: str, prompt: str):
    """
    отправляет запрос в наше API и пытается вытащить вердикт (0 или 1).
    """
    url = "http://localhost:8000/predict"
    
    try:
        req = requests.post(url, json={"system_prompt": prompt, "text": text})
        ans = req.json().get("response", "")
        
        #print("DEBUG:", ans)
        
        match = re.search(r'\{.*\}', ans.replace('\n', ' '))
        if match:
            parsed = json.loads(match.group(0))
            return int(parsed.get("verdict", 0))
        return 0 # если не смогли распарсить, считаем что не спам
    except:
        return 0

def run_tests():
    df = load_data("spam.csv")
    prompts = get_prompts()
    
    results =[]
    
    for name, prompt_text in prompts.items():
        print(f"\n--- Тестируем {name} ---")
        y_true = df['target'].tolist()
        y_pred = []
        
        for text in tqdm(df['text']):
            pred = get_verdict_from_llm(text, prompt_text)
            y_pred.append(pred)
            
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        results.append({
            "Method": name,
            "Accuracy": round(acc, 3),
            "Precision": round(prec, 3),
            "Recall": round(rec, 3),
            "F1": round(f1, 3)
        })
        
    df_res = pd.DataFrame(results)
    print("\nИТОГОВЫЙ ОТЧЕТ:\n")
    print(df_res)

if __name__ == "__main__":
    run_tests()
