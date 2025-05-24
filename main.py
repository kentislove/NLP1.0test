import os
import requests
import cohere
import numpy as np

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ====== API KEY 設定 ======
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

# ====== FastAPI & CORS ======
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Perplexity 純問答服務 ======
@app.post("/ask")
async def ask(request: Request):
    """
    使用 Perplexity API 回答用戶問題
    Request JSON: {"message": "你的問題"}
    Response JSON: {"reply": "回答內容"}
    """
    data = await request.json()
    question = data.get("message", "")
    if not question:
        return JSONResponse({"reply": "請提供問題內容"}, status_code=400)

    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "sonar-medium-online",
        "messages": [
            {"role": "system", "content": "Be precise and concise."},
            {"role": "user", "content": question}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }

    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    content = resp.json()
    reply = content.get("choices", [])[0].get("message", {}).get("content", "")
    return JSONResponse({"reply": reply})

# ====== Cohere FAQ 語意檢索 ======
FAQ_FILE = os.path.join(os.path.dirname(__file__), "docs", "faq.txt")
co = cohere.Client(COHERE_API_KEY)


def load_faqs():
    """
    從 FAQ 文件讀取問答，格式: 問題|答案
    """
    faqs = []
    if not os.path.exists(FAQ_FILE):
        print(f"FAQ 檔案不存在: {FAQ_FILE}")
        return faqs

    with open(FAQ_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|", 1)
            if len(parts) == 2:
                faqs.append({"question": parts[0], "answer": parts[1]})
    return faqs

faqs = load_faqs()
faq_questions = [faq["question"] for faq in faqs]
faq_embeddings = []
if faq_questions:
    resp = co.embed(texts=faq_questions, model="embed-english-v3.0")
    faq_embeddings = resp.embeddings

@app.post("/faq")
async def faq(request: Request):
    """
    使用 Cohere 向量搜尋匹配最相似 FAQ
    Request JSON: {"message": "你的問題"}
    Response JSON: {"reply": "匹配的回答"}
    """
    data = await request.json()
    question = data.get("message", "")
    if not question:
        return JSONResponse({"reply": "請提供問題內容"}, status_code=400)
    if not faq_embeddings:
        return JSONResponse({"reply": "FAQ 資料尚未載入"}, status_code=500)

    q_resp = co.embed(texts=[question], model="embed-english-v3.0")
    q_emb = q_resp.embeddings[0]

    # 計算餘弦相似度並選出最高
    sims = [
        np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb))
        for emb in faq_embeddings
    ]
    best_idx = int(np.argmax(sims))
    answer = faqs[best_idx]["answer"]
    return JSONResponse({"reply": answer})

# ====== 其他原有 RAG 問答、Webhook、Gradio 等可照舊加入 ======
@app.get("/")
async def root():
    return {"status": "running"}
