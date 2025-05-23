import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import gradio as gr
from typing import List, Tuple
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from utils import load_documents_from_folder

# ====== 設定區 ======
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# 相似度門檻，可透過環境變數調整，預設 0.9
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.9"))

# Local paths
VECTOR_STORE_PATH = "./faiss_index"
DOCUMENTS_PATH = "./docs"

# 確保資料夾存在
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(DOCUMENTS_PATH, exist_ok=True)

# ====== 嵌入模型與 LLM ======
# 使用 OpenAI 文字嵌入模型（最便宜的 text-embedding-ada-002）
embedding_model = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-ada-002"
)
# 使用最便宜的 chat 模型（gpt-3.5-turbo）作為 fallback
openai_llm = OpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-3.5-turbo",
    temperature=0.3
)

vectorstore: FAISS = None

# ====== Vector Store 構建 ======
def build_vector_store() -> FAISS:
    documents = load_documents_from_folder(DOCUMENTS_PATH)
    if not documents:
        raise RuntimeError(
            "docs 資料夾內沒有可用文件，請至少放入一份 txt/pdf/doc/docx/xls/xlsx/csv 檔案！"
        )
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = splitter.split_documents(documents)
    faiss_db = FAISS.from_documents(texts, embedding_model)
    faiss_db.save_local(VECTOR_STORE_PATH)
    return faiss_db

# 確保 vectorstore 已載入或已建構
def ensure_vectorstore():
    global vectorstore
    if vectorstore is None:
        if os.path.exists(VECTOR_STORE_PATH):
            vectorstore = FAISS.load_local(
                VECTOR_STORE_PATH,
                embedding_model,
                allow_dangerous_deserialization=True
            )
        else:
            vectorstore = build_vector_store()

# ====== 問答主邏輯 ======
def rag_answer(question: str) -> str:
    # 向量相似度檢索
    ensure_vectorstore()
    try:
        docs_and_scores: List[Tuple] = vectorstore.similarity_search_with_score(
            question, k=1
        )
    except Exception:
        # 索引不存在或異常，重建後再試一次
        vectorstore = build_vector_store()
        docs_and_scores = vectorstore.similarity_search_with_score(
            question, k=1
        )

    if docs_and_scores:
        doc, score = docs_and_scores[0]
        # FAISS L2 distance 轉成相似度分數 (越接近 1 越相似)
        similarity = 1 / (1 + score)
        if similarity >= SIMILARITY_THRESHOLD:
            return doc.page_content

    # 若相似度未達門檻，使用 LLM 回答
    return openai_llm(question)

# ====== FastAPI 與 Gradio 啟動區 ======
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
    body = await request.json()
    user_message = body.get("message", "")
    reply = rag_answer(user_message)
    return JSONResponse({"reply": reply})

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 智能問答機器人")
    with gr.Row():
        with gr.Column():
            qbox = gr.Textbox(label="請輸入問題")
            abox = gr.Textbox(label="回答")
            btn = gr.Button("送出")
            btn.click(fn=rag_answer, inputs=qbox, outputs=abox)

# 將 Gradio 掛載到 FastAPI
from gradio.routes import mount_gradio_app
app = mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
