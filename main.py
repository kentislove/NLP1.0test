import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import gradio as gr
from typing import List, Tuple
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from utils import load_documents_from_folder

# LINE Bot SDK
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

# ====== 環境變數設定 ======
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.9"))

# ====== 模型與向量索引設定 ======
# 嵌入模型：text-embedding-ada-002
embedding_model = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-3-small"
)
# 回退 LLM：gpt-3.5-turbo
openai_llm = OpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-3.5-turbo",
    temperature=0.3
)
VECTOR_STORE_PATH = "./faiss_index"
DOCUMENTS_PATH = "./docs"

# 初始化 LINE Bot
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# 確保資料目錄與索引資料夾
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(DOCUMENTS_PATH, exist_ok=True)
vectorstore: FAISS = None

# ====== 向量索引建立 ======
def build_vector_store() -> FAISS:
    docs = load_documents_from_folder(DOCUMENTS_PATH)
    if not docs:
        raise RuntimeError("docs 資料夾內至少要有一個 txt 檔！")
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = splitter.split_documents(docs)
    faiss_db = FAISS.from_documents(texts, embedding_model)
    faiss_db.save_local(VECTOR_STORE_PATH)
    return faiss_db

# 載入或建立索引
def ensure_vectorstore():
    global vectorstore
    if not os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
        print("未找到 FAISS 索引，開始建立...")
        vectorstore = build_vector_store()
    else:
        print("載入現有 FAISS 索引...")
        vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embedding_model)

# ====== RAG 問答邏輯 ======
def rag_answer(question: str) -> str:
    ensure_vectorstore()
    try:
        docs_and_scores: List[Tuple] = vectorstore.similarity_search_with_score(question, k=1)
    except Exception:
        # 重建索引後再次嘗試
        vectorstore = build_vector_store()
        docs_and_scores = vectorstore.similarity_search_with_score(question, k=1)
    if docs_and_scores:
        doc, score = docs_and_scores[0]
        similarity = 1 / (1 + score)
        if similarity >= SIMILARITY_THRESHOLD:
            return doc.page_content
    return openai_llm(question)

# ====== FastAPI + LINE + Gradio ======
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

# LINE Webhook 回調
@app.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body = (await request.body()).decode()
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    return "OK"

# LINE 訊息處理
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event: MessageEvent):
    user_text = event.message.text
    reply = rag_answer(user_text)
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply)
    )

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 智能問答機器人")
    with gr.Row():
        with gr.Column():
            qbox = gr.Textbox(label="請輸入問題")
            abox = gr.Textbox(label="回答")
            btn = gr.Button("送出")
            btn.click(fn=rag_answer, inputs=qbox, outputs=abox)

from gradio.routes import mount_gradio_app
app = mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
