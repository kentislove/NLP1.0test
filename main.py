import os
import requests
import cohere
import numpy as np

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import gradio as gr
from gradio.routes import mount_gradio_app

from typing import List, Tuple

# LangChain for RAG
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from utils import load_documents_from_folder

# LINE Bot SDK
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

# ====== 環境變數設定 ======
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.9"))
VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), "faiss_index")
DOCUMENTS_PATH = os.path.join(os.path.dirname(__file__), "docs")
FAQ_FILE = os.path.join(DOCUMENTS_PATH, "faq.txt")

# ====== FastAPI & CORS ======
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Mount Gradio 靜態檔案
app.mount(
    "/gradio/static",
    StaticFiles(directory=os.path.join(os.path.dirname(gr.__file__), "static")),
    name="gradio-static"
)

# ====== Perplexity 純問答服務 ======
@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = data.get("message", "").strip()
    if not question:
        return JSONResponse({"reply": "請提供問題內容"}, status_code=400)

    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
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
    choices = resp.json().get("choices", [])
    reply = choices[0].get("message", {}).get("content", "") if choices else ""
    return JSONResponse({"reply": reply})

# ====== Cohere FAQ 語意檢索 ======
co = cohere.Client(COHERE_API_KEY)


def load_faqs():
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
    data = await request.json()
    question = data.get("message", "").strip()
    if not question:
        return JSONResponse({"reply": "請提供問題內容"}, status_code=400)
    if not faq_embeddings:
        return JSONResponse({"reply": "FAQ 資料尚未載入"}, status_code=500)
    q_resp = co.embed(texts=[question], model="embed-english-v3.0")
    q_emb = q_resp.embeddings[0]
    sims = [np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb)) for emb in faq_embeddings]
    idx = int(np.argmax(sims))
    return JSONResponse({"reply": faqs[idx]["answer"]})

# ====== RAG 向量檢索服務 ======
# 設定嵌入模型與 LLM
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
openai_llm = OpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo", temperature=0.3)

os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(DOCUMENTS_PATH, exist_ok=True)
vectorstore: FAISS

def build_vector_store() -> FAISS:
    docs = load_documents_from_folder(DOCUMENTS_PATH)
    if not docs:
        raise RuntimeError("docs 資料夾內至少要有一個 txt 檔！")
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = splitter.split_documents(docs)
    db = FAISS.from_documents(texts, embedding_model)
    db.save_local(VECTOR_STORE_PATH)
    return db

# 確保索引
try:
    if not os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
        vectorstore = build_vector_store()
    else:
        vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embedding_model)
except Exception:
    vectorstore = build_vector_store()

# 問答函式，包含 global 聲明在首行
def rag_answer(question: str) -> str:
    global vectorstore
    try:
        docs_and_scores = vectorstore.similarity_search_with_score(question, k=1)
    except Exception:
        vectorstore = build_vector_store()
        docs_and_scores = vectorstore.similarity_search_with_score(question, k=1)
    if docs_and_scores:
        doc, score = docs_and_scores[0]
        similarity = 1 / (1 + score)
        if similarity >= SIMILARITY_THRESHOLD:
            return doc.page_content
    return openai_llm(question)

# ====== LINE Bot Webhook ======
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

@app.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body = (await request.body()).decode()
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event: MessageEvent):
    user_text = event.message.text
    reply = rag_answer(user_text)
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

# ====== Gradio Web UI ======
demo = gr.Blocks()
with demo:
    gr.Markdown("# 智能問答機器人")
    with gr.Row():
        qbox = gr.Textbox(label="請輸入問題")
        abox = gr.Textbox(label="回答")
        btn = gr.Button("送出")
        btn.click(fn=rag_answer, inputs=qbox, outputs=abox)
# 挂載到 /gradio
app = mount_gradio_app(app, demo, path="/gradio")

# ====== 根路由 ======
@app.get("/")
async def root():
    return {"status": "running"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
