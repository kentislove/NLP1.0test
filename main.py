# main_function_router.py (支援多 Index FAISS + 記憶體監控/清理)

import os
import psutil
import time
import gc
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import gradio as gr
from typing import List, Tuple, Dict
import re
import requests

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from utils import load_documents_from_folder

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))

embedding_model_name = "johnpaulbin/bge-m3-distilled-tiny"
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
VECTOR_STORE_PATH = "./faiss_index"
DOCUMENTS_PATH = "./docs"

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(DOCUMENTS_PATH, exist_ok=True)

vectorstore_cache: Dict[str, FAISS] = {}  # 支援多 index 載入
vectorstore_last_used: Dict[str, float] = {}  # 最後使用時間紀錄
MAX_IDLE_SECONDS = 200  # index 閒置超過此秒數將被清除

# ========== 加入記憶體使用紀錄函數 ==========
def log_memory_usage(note=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024
    print(f"[MEMORY] {note} - RSS: {mem:.2f} MB")

# ========== 清除過期的 FAISS index ==========
def clear_expired_indexes():
    now = time.time()
    expired = [key for key, t in vectorstore_last_used.items() if now - t > MAX_IDLE_SECONDS]
    for key in expired:
        del vectorstore_cache[key]
        del vectorstore_last_used[key]
        gc.collect()
        print(f"[GC] 已清除 index: {key}")

# ========== 延遲載入多組 FAISS ==========
def lazy_load_vectorstore(domain: str) -> FAISS:
    clear_expired_indexes()
    if domain in vectorstore_cache:
        vectorstore_last_used[domain] = time.time()
        return vectorstore_cache[domain]
    index_path = os.path.join(VECTOR_STORE_PATH, domain)
    if os.path.exists(os.path.join(index_path, "index.faiss")) and \
       os.path.exists(os.path.join(index_path, "index.pkl")):
        try:
            vs = FAISS.load_local(index_path, embedding_model)
            vectorstore_cache[domain] = vs
            vectorstore_last_used[domain] = time.time()
            log_memory_usage(f"FAISS loaded for {domain}")
            return vs
        except Exception as e:
            print(f"[ERROR] 載入 {domain} index 失敗: {e}")
            return None
    else:
        print(f"[WARNING] 找不到 index: {domain}")
        return None

# ========== 問答邏輯（指定 domain） ==========
def rag_answer(question: str, domain: str = "default") -> str:
    vs = lazy_load_vectorstore(domain)
    if not vs:
        return "抱歉，目前知識庫尚未就緒。"
    try:
        docs_and_scores: List[Tuple] = vs.similarity_search_with_score(question, k=1)
        if docs_and_scores:
            doc, score = docs_and_scores[0]
            similarity = 1 / (1 + score)
            if similarity >= SIMILARITY_THRESHOLD:
                return doc.page_content
            else:
                return "這部分我不確定，建議嘗試其他問題喔！"
    except Exception as e:
        return f"知識查詢失敗：{str(e)}"

# ========== Function Routing ==========
def route_message(text: str) -> str:
    log_memory_usage("Before routing")
    t = text.lower()
    if "翻譯" in t or t.startswith("translate"):
        return translate_text(text)
    elif re.search(r"\d+[\+\-\*/]", t) or any(k in t for k in ["多少", "計算", "sqrt", "公式"]):
        return ask_wolfram(text)
    elif any(q in t for q in ["什麼", "為什麼", "who", "what", "when", "how"]):
        return search_serper(text)
    elif any(k in t for k in ["圖片", "風景", "找圖"]):
        return search_image(text)
    else:
        return rag_answer(text, domain="default")

# ========== 外部 API Functions ==========
def ask_wolfram(query: str) -> str:
    APPID = os.getenv("WOLFRAM_APPID", "")
    url = f"https://api.wolframalpha.com/v1/result?i={requests.utils.quote(query)}&appid={APPID}"
    try:
        r = requests.get(url, timeout=5)
        return r.text if r.status_code == 200 else "Wolfram 查詢失敗"
    except:
        return "無法連線 Wolfram Alpha"

def translate_text(text: str) -> str:
    try:
        payload = {"q": text, "source": "auto", "target": "en"}
        r = requests.post("https://libretranslate.de/translate", json=payload, timeout=5)
        return r.json().get("translatedText", "翻譯失敗")
    except:
        return "翻譯服務無法連線"

def search_serper(query: str) -> str:
    try:
        api_key = os.getenv("SERPER_API_KEY", "")
        r = requests.post("https://google.serper.dev/search",
                         headers={"X-API-KEY": api_key},
                         json={"q": query})
        result = r.json()
        if result.get("answerBox"):
            return result["answerBox"].get("answer", "找不到解答")
        elif result.get("organic"):
            top = result["organic"][0]
            return f"{top['title']}\n{top['snippet']}\n{top['link']}"
        else:
            return "沒有找到明確結果"
    except:
        return "搜尋失敗或 API 限額已達"

def search_image(keyword: str) -> str:
    try:
        api_key = os.getenv("PEXELS_API_KEY", "")
        r = requests.get("https://api.pexels.com/v1/search",
                         headers={"Authorization": api_key},
                         params={"query": keyword, "per_page": 1})
        data = r.json()
        if data["photos"]:
            return data["photos"][0]["url"]
        return "找不到圖片"
    except:
        return "圖片搜尋失敗"

# ========== FastAPI + LINE Bot ==========
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
    reply = route_message(user_text)
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply)
    )

with gr.Blocks() as demo:
    gr.Markdown("# AI Bot (Function Calling + FAISS + Monitor)")
    with gr.Row():
        with gr.Column():
            qbox = gr.Textbox(label="請輸入問題")
            abox = gr.Textbox(label="回答", interactive=False)
            btn = gr.Button("送出")
            btn.click(fn=route_message, inputs=qbox, outputs=abox)

from gradio.routes import mount_gradio_app
app = mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    log_memory_usage("啟動前")
    uvicorn.run("main_function_router:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")), reload=False)
