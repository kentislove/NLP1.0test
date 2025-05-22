import os
import shutil
import gradio as gr
import json
import hmac
import hashlib
import base64
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse, FileResponse
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from utils import load_documents_from_folder
from typing import List
import httpx
import datetime
import time

# ===== 參數控制 =====
BATCH_SIZE = 20
WAIT_SEC = 8      # 可依實際頻寬/錯誤頻率改大，如 8~15

# ===== 對話紀錄函式 =====
LOG_FILE = "conversation_logs.txt"
def log_conversation(source, user_id, question, answer):
    log_entry = (
        f"{datetime.datetime.now().isoformat()} "
        f"[{source}] "
        f"({user_id})\n"
        f"Q: {question}\n"
        f"A: {answer}\n"
        f"{'-'*40}\n"
    )
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry)

# ===== Cohere & 向量庫設定 =====
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

VECTOR_STORE_PATH = "./faiss_index"
DOCUMENTS_PATH = "./docs"
DOCS_STATE_PATH = os.path.join(VECTOR_STORE_PATH, "last_docs.json")
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(DOCUMENTS_PATH, exist_ok=True)

embedding_model = CohereEmbeddings(
    cohere_api_key=COHERE_API_KEY,
    model="embed-multilingual-v3.0"
)
llm = ChatCohere(
    cohere_api_key=COHERE_API_KEY,
    model="command-r-plus",
    temperature=0.3
)
vectorstore = None
qa = None

def get_uploaded_files_list():
    files = os.listdir(DOCUMENTS_PATH)
    return "\n".join(files) if files else "(目前沒有檔案)"

def get_current_docs_state() -> dict:
    docs_state = {}
    for f in os.listdir(DOCUMENTS_PATH):
        path = os.path.join(DOCUMENTS_PATH, f)
        if os.path.isfile(path):
            docs_state[f] = os.path.getmtime(path)
    return docs_state

def load_last_docs_state() -> dict:
    if os.path.exists(DOCS_STATE_PATH):
        with open(DOCS_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_docs_state(state: dict):
    with open(DOCS_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f)

def get_new_or_updated_files(current: dict, last: dict) -> List[str]:
    changed = []
    for name, mtime in current.items():
        if name not in last or last[name] < mtime:
            changed.append(name)
    return changed

def batch_vectorize(texts, embedding_model, batch_size=BATCH_SIZE, wait_sec=WAIT_SEC):
    all_texts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"向量化：第 {i//batch_size+1} 批，共 {len(batch)} 筆 ...")
        all_texts.extend(batch)
        if i + batch_size < len(texts):
            print(f"sleep {wait_sec}s ...")
            time.sleep(wait_sec)
    return all_texts


def build_vector_store(docs_state: dict = None):
    documents = load_documents_from_folder(DOCUMENTS_PATH)
    splitter = CharacterTextSplitter(
        separator='',        # 直接按字元硬切
        chunk_size=500,
        chunk_overlap=10,
    )
    texts = splitter.split_documents(documents)
    if not texts:
        raise RuntimeError("docs 資料夾內沒有可用文件，無法建立向量資料庫，請至少放入一份 txt/pdf/doc/docx/xls/xlsx/csv 檔案！")
    # 分批向量化
    print(f"分批向量化所有文件：{len(texts)} 筆，每批 {BATCH_SIZE}，間隔 {WAIT_SEC} 秒")
    FAISS.from_documents(batch_vectorize(texts, embedding_model), embedding_model).save_local(VECTOR_STORE_PATH)
    if docs_state:
        save_docs_state(docs_state)
    return FAISS.load_local(VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)

def add_new_files_to_vector_store(db, new_files: List[str], docs_state: dict):
    from langchain_community.document_loaders import (
        TextLoader,
        UnstructuredPDFLoader,
        UnstructuredWordDocumentLoader,
        UnstructuredExcelLoader
    )
    splitter = CharacterTextSplitter(
        separator='',        # 直接按字元硬切
        chunk_size=500,
        chunk_overlap=10,
    )
    new_documents = []
    for file in new_files:
        filepath = os.path.join(DOCUMENTS_PATH, file)
        if os.path.isfile(filepath):
            ext = os.path.splitext(file)[1].lower()
            if ext == ".txt":
                loader = TextLoader(filepath, autodetect_encoding=True)
                docs = loader.load()
            elif ext == ".pdf":
                loader = UnstructuredPDFLoader(filepath)
                docs = loader.load()
            elif ext in [".doc", ".docx"]:
                loader = UnstructuredWordDocumentLoader(filepath)
                docs = loader.load()
            elif ext in [".xlsx", ".xls"]:
                loader = UnstructuredExcelLoader(filepath)
                docs = loader.load()
            elif ext == ".csv":
                from utils import parse_csv_file
                docs = parse_csv_file(filepath)
            else:
                print(f"不支援的格式：{file}")
                continue
            new_documents.extend(docs)
    if new_documents:
        texts = splitter.split_documents(new_documents)
        # 分批處理
        print(f"分批向量化新增文件：{len(texts)} 筆，每批 {BATCH_SIZE}，間隔 {WAIT_SEC} 秒")
        batched_texts = batch_vectorize(texts, embedding_model)
        db.add_documents(batched_texts)
        db.save_local(VECTOR_STORE_PATH)
        save_docs_state(docs_state)
    return db

def ensure_qa():
    global vectorstore, qa
    current_docs_state = get_current_docs_state()
    last_docs_state = load_last_docs_state()
    if vectorstore is None:
        if os.path.exists(VECTOR_STORE_PATH) and os.listdir(VECTOR_STORE_PATH):
            vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)
            new_files = get_new_or_updated_files(current_docs_state, last_docs_state)
            if new_files:
                print(f"偵測到新文件或文件變動：{new_files}，自動增量加入向量庫！")
                vectorstore = add_new_files_to_vector_store(vectorstore, new_files, current_docs_state)
            elif len(current_docs_state) != len(last_docs_state):
                print(f"文件數變動，重新建構向量庫")
                vectorstore = build_vector_store(current_docs_state)
        else:
            vectorstore = build_vector_store(current_docs_state)
    else:
        new_files = get_new_or_updated_files(current_docs_state, last_docs_state)
        if new_files:
            print(f"偵測到新文件或文件變動：{new_files}，自動增量加入向量庫！")
            vectorstore = add_new_files_to_vector_store(vectorstore, new_files, current_docs_state)
        elif len(current_docs_state) != len(last_docs_state):
            print(f"文件數變動，重新建構向量庫")
            vectorstore = build_vector_store(current_docs_state)

    if qa is None:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

def limit_answer(answer, max_paragraphs=3, max_chars=100):
    # 切分段落，只保留前 max_paragraphs 段
    paras = [p for p in answer.split('\n') if p.strip()]
    limited = '\n'.join(paras[:max_paragraphs])
    # 若總長度超過 max_chars，就截斷
    if len(limited) > max_chars:
        limited = limited[:max_chars].rstrip()
    return limited
    
def rag_answer(question):
    ensure_qa()
    answer = qa.run(question)
    return limit_answer(answer, max_paragraphs=3, max_chars=100)
    
def handle_upload(files, progress=gr.Progress()):
    new_files = []
    total = len(files)
    for idx, file in enumerate(files):
        filename = os.path.basename(file.name)
        dest = os.path.join(DOCUMENTS_PATH, filename)
        shutil.copyfile(file.name, dest)
        new_files.append(filename)
        progress((idx + 1) / total, desc=f"複製檔案：{filename}")
    progress(0.8, desc="正在匯入向量庫 ...")
    current_docs_state = get_current_docs_state()
    last_docs_state = load_last_docs_state()
    db = None
    if os.path.exists(VECTOR_STORE_PATH) and os.listdir(VECTOR_STORE_PATH):
        db = FAISS.load_local(VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)
    else:
        db = build_vector_store(current_docs_state)
        return (f"向量庫首次建立完成，共匯入 {len(new_files)} 檔案。", get_uploaded_files_list())
    add_new_files_to_vector_store(db, new_files, current_docs_state)
    progress(1.0, desc="完成！")
    return (f"成功匯入 {len(new_files)} 檔案並加入向量庫：{', '.join(new_files)}", get_uploaded_files_list())

# ===== Gradio UI 主程式（新版版面配置） =====
with gr.Blocks() as demo:
    gr.Markdown("# Cohere 向量檢索問答機器人")
    with gr.Row():
        with gr.Column(scale=2):
            user_name = gr.Textbox(label="你的名字/暱稱", placeholder="可空白，預設anonymous")
            question_box = gr.Textbox(label="輸入問題", placeholder="請輸入問題")
            submit_btn = gr.Button("送出")
            answer_box = gr.Textbox(label="AI 回答")
        with gr.Column(scale=1):
            upload_box = gr.File(
                label="上傳新文件（支援 doc/docx/xls/xlsx/pdf/txt/csv，多選）",
                file_count="multiple",
                file_types=[".doc", ".docx", ".xls", ".xlsx", ".pdf", ".txt", ".csv"]
            )
            upload_btn = gr.Button("匯入並轉換成向量資料庫")
            status_box = gr.Textbox(label="操作狀態/訊息")
            file_list_box = gr.Textbox(label="已上傳檔案清單", value=get_uploaded_files_list(), interactive=False, lines=8)
    upload_btn.click(
        fn=handle_upload,
        inputs=[upload_box],
        outputs=[status_box, file_list_box]
    )
    # 新增 user_name 輸入
    def gradio_qa(name, question):
        answer = rag_answer(question)
        log_conversation("gradio", name or "anonymous", question, answer)
        return answer

    submit_btn.click(fn=gradio_qa, inputs=[user_name, question_box], outputs=answer_box)

# ===== FastAPI 設定 =====
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from gradio.routes import mount_gradio_app
app = mount_gradio_app(app, demo, path="/gradio")

@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def root():
    return {"status": "running"}

@app.post("/webhook")
async def webhook(request: Request):
    payload = await request.json()
    user_message = payload.get("message", "")
    client_ip = request.client.host
    answer = rag_answer(user_message)
    log_conversation("webhook", client_ip, user_message, answer)
    return {"reply": answer}

# ===== LINE BOT 設定 =====
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_API_REPLY_URL = "https://api.line.me/v2/bot/message/reply"

def verify_line_signature(request_body: bytes, x_line_signature: str) -> bool:
    hash = hmac.new(LINE_CHANNEL_SECRET.encode('utf-8'), request_body, hashlib.sha256).digest()
    signature = base64.b64encode(hash)
    return hmac.compare_digest(signature, x_line_signature.encode('utf-8'))

@app.post("/line/webhook")
async def line_webhook(request: Request, x_line_signature: str = Header(None)):
    body = await request.body()
    if not verify_line_signature(body, x_line_signature):
        raise HTTPException(status_code=400, detail="Invalid signature")
    payload = await request.json()
    events = payload.get("events", [])
    for event in events:
        if event["type"] == "message" and event["message"]["type"] == "text":
            user_text = event["message"]["text"]
            reply_token = event["replyToken"]
            line_user_id = event["source"]["userId"]
            answer = rag_answer(user_text)
            log_conversation("line", line_user_id, user_text, answer)
            headers = {
                "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
                "Content-Type": "application/json"
            }
            data = {
                "replyToken": reply_token,
                "messages": [
                    {"type": "text", "text": answer}
                ]
            }
            async with httpx.AsyncClient() as client:
                await client.post(LINE_API_REPLY_URL, headers=headers, json=data)
    return JSONResponse(content={"status": "ok"})

# ===== 下載對話歷史 API =====
@app.get("/download_logs")
async def download_logs():
    return FileResponse(LOG_FILE, media_type="text/plain", filename="conversation_logs.txt")
