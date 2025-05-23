import os
import shutil
import gradio as gr
import json
import hmac
import hashlib
import base64
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from utils import load_documents_from_folder
from typing import List
import httpx
import time

# ====== 設定區 ======
DEBUG_MODE = os.getenv("DEBUG_MODE", "0") == "1"
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
VECTOR_STORE_PATH = "./faiss_index"
DOCUMENTS_PATH = "./docs"
DOCS_STATE_PATH = os.path.join(VECTOR_STORE_PATH, "last_docs.json")
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(DOCUMENTS_PATH, exist_ok=True)

# ====== 模型建構 ======
if DEBUG_MODE:
    print("[DEBUG] 啟用模擬模式，將不呼叫 Cohere API。")
    class DummyEmbedding:
        def embed_documents(self, docs): return [[0.1] * 768 for _ in docs]
        def embed_query(self, query): return [0.1] * 768
    class DummyLLM:
        def invoke(self, prompt): return "（模擬精煉）"
        def __call__(self, *args, **kwargs): return "（模擬回答）"
    embedding_model = DummyEmbedding()
    llm = DummyLLM()
else:
    embedding_model = CohereEmbeddings(cohere_api_key=COHERE_API_KEY, model="embed-multilingual-v3.0")
    llm = ChatCohere(cohere_api_key=COHERE_API_KEY, model="command-r", temperature=0.3)

vectorstore = None
qa = None

# ====== 問題精煉與分類 ======
def refine_question(question: str) -> str:
    if DEBUG_MODE:
        keywords = ["行程", "價格", "費用", "地點", "日期", "旅遊", "報名", "住宿", "機票"]
        for word in keywords:
            if word in question:
                return word
        return "歡樂旅遊"
    else:
        from transformers import pipeline
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        try:
            summary = summarizer(question, max_length=15, min_length=5, do_sample=False)
            return summary[0]["summary_text"]
        except Exception:
            return "歡樂旅遊"

def classify_question_intent(question: str) -> str:
    if any(x in question for x in ["多少", "價格", "費用", "價錢"]): return "詢價"
    elif any(x in question for x in ["行程", "去哪", "路線"]): return "查詢行程"
    elif any(x in question for x in ["訂位", "預約", "報名"]): return "預約"
    else: return "一般問題"

# ====== QA主邏輯 ======
def get_uploaded_files_list():
    files = os.listdir(DOCUMENTS_PATH)
    return "\n".join(files) if files else "(目前沒有檔案)"

def get_current_docs_state() -> dict:
    return {f: os.path.getmtime(os.path.join(DOCUMENTS_PATH, f)) for f in os.listdir(DOCUMENTS_PATH)}

def load_last_docs_state() -> dict:
    if os.path.exists(DOCS_STATE_PATH):
        with open(DOCS_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_docs_state(state: dict):
    with open(DOCS_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f)

def get_new_or_updated_files(current: dict, last: dict) -> List[str]:
    return [name for name, mtime in current.items() if name not in last or last[name] < mtime]

def build_vector_store(docs_state: dict = None):
    documents = load_documents_from_folder(DOCUMENTS_PATH)
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = splitter.split_documents(documents)
    if not texts:
        raise RuntimeError("docs 資料夾內沒有可用文件，請至少放入一份 txt/pdf/doc/docx/xls/xlsx/csv 檔案！")
    faiss_db = FAISS.from_documents(texts, embedding_model)
    faiss_db.save_local(VECTOR_STORE_PATH)
    if docs_state: save_docs_state(docs_state)
    return faiss_db

def ensure_qa():
    global vectorstore, qa
    current_docs_state = get_current_docs_state()
    last_docs_state = load_last_docs_state()
    if vectorstore is None:
        if os.path.exists(VECTOR_STORE_PATH):
            vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)
        else:
            vectorstore = build_vector_store(current_docs_state)
    if qa is None:
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

def rag_answer(question):
    ensure_qa()
    answer = qa.run(question)
    if len(answer) > 50:
        refined = refine_question(question)
        intent = classify_question_intent(question)
        url = f"https://www.google.com/search?q={refined}"
        return f"此問題屬於「{intent}」類型，建議參考：{url}\n\n可直接聯絡我們歡樂旅遊，Email：service@happytravel.com，電話：02-1234-5678"
    return f"{answer}\n\n可直接聯絡我們歡樂旅遊，Email：service@happytravel.com，電話：02-1234-5678"

# ====== FastAPI 啟動區 ======
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "running"}

@app.post("/webhook")
async def webhook(request: Request):
    payload = await request.json()
    user_message = payload.get("message", "")
    reply = rag_answer(user_message)
    return {"reply": reply}

# ====== 啟動 Gradio 並掛載到 FastAPI ======
with gr.Blocks() as demo:
    gr.Markdown("# 歡樂旅遊 智能問答機器人")
    with gr.Row():
        with gr.Column():
            qbox = gr.Textbox(label="請輸入問題")
            abox = gr.Textbox(label="AI 回答")
            btn = gr.Button("送出")
            btn.click(fn=rag_answer, inputs=qbox, outputs=abox)

from gradio.routes import mount_gradio_app
app = mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)  # reload=False 可避免 Render 問題
