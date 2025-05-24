import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import gradio as gr
from typing import List, Tuple
# from langchain.embeddings.openai import OpenAIEmbeddings # 移除 OpenAI Embedding
# from langchain.llms import OpenAI # 移除 OpenAI LLM
# from langchain.llms import Cohere # 移除 Cohere LLM

# 改用開源嵌入模型
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from utils import load_documents_from_folder

# LINE Bot SDK
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

# ====== 環境變數設定 ======
# 不再需要 OPENAI_API_KEY 和 COHERE_API_KEY
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75")) # 調整相似度閾值，0.75-0.85 較常見，可依需求調整

# ====== 模型與向量索引設定 ======
# 改用 HuggingFace 的開源嵌入模型，例如 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
# 這個模型支援多國語言，且體積較小，適合部署。
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

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
        raise RuntimeError(f"'{DOCUMENTS_PATH}' 資料夾內至少要有一個 txt 或其他支援的檔！")
    # 調整 chunk_size 和 chunk_overlap 參數以適應您的文檔內容和期望的片段大小
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
        try:
            vectorstore = build_vector_store()
            print("FAISS 索引建立完成。")
        except RuntimeError as e:
            print(f"建立 FAISS 索引失敗: {e}")
            vectorstore = None
    else:
        print("載入現有 FAISS 索引...")
        try:
            vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embedding_model)
            print("FAISS 索引載入完成。")
        except Exception as e:
            print(f"載入 FAISS 索引失敗: {e}")
            vectorstore = None

# ====== FAISS RAG (純檢索) 問答邏輯 ======
def rag_answer(question: str) -> str:
    ensure_vectorstore()

    if vectorstore:
        try:
            # 使用 cosine distance，分數越低代表越相似。
            # FAISS 的 similarity_search_with_score 返回的是 L2 distance (歐幾里得距離)，分數越低越相似。
            # 這裡我們將它轉換為一個 0-1 範圍的相似度，方便閾值判斷。
            # 轉換公式 `1 / (1 + distance)` 是一個簡單的非線性轉換，將距離映射到相似度。
            # 如果使用 cosine similarity，則直接取相似度，值介於 -1 到 1，越接近 1 越相似。
            # HuggingFaceEmbeddings 預設使用 cosine similarity。
            docs_and_scores: List[Tuple] = vectorstore.similarity_search_with_score(question, k=1)
            
            if docs_and_scores:
                doc, score = docs_and_scores[0]
                # HuggingFaceEmbeddings 通常返回的是 L2 distance (距離)，所以距離越小越相似。
                # 我們需要確保 SIMILARITY_THRESHOLD 是針對距離的。
                # 或者可以將距離轉換為相似度 (例如 1 / (1 + distance) 或 (max_dist - dist) / max_dist )
                # 這裡假設 score 是距離，我們希望它小於某個閾值。
                # 如果您的 embedding_model 返回的是相似度，則需要將 SIMILARITY_THRESHOLD 的判斷方向反過來。
                
                # 由於HuggingFaceEmbeddings搭配FAISS預設是L2距離，距離越小代表越相似。
                # 因此我們的SIMILARITY_THRESHOLD應該設定為一個「最大允許距離」，而不是相似度。
                # 這裡為了維持原有的 'similarity' 概念，我們做一個簡單的轉換，但請注意其非線性。
                distance = score
                # 簡單轉換為一個非線性的「相似度」概念，讓它介於 0 到 1
                # 距離越小，相似度越接近 1
                # 這裡的 SIMILARITY_THRESHOLD 還是要對應到轉換後的「相似度」值。
                # 若您想直接判斷距離，則 SIMILARITY_THRESHOLD 應該設為距離的最大值。
                # 例如，如果距離小於 0.5 才算匹配，SIMILARITY_THRESHOLD = 0.5
                
                # 為了避免混淆，我們直接判斷距離，並調整 SIMILARITY_THRESHOLD 的意義
                # 將 SIMILARITY_THRESHOLD 設為「最大距離」，距離越小越相似
                # 原始程式碼的 SIMILARITY_THRESHOLD 被解釋為「相似度要大於此值」，因此我反過來判斷
                # 如果是 L2 距離，分數越小越好，因此我們需要一個 `max_distance_threshold`
                # 這裡保持原始的 SIMILARITY_THRESHOLD 變數名，但其含義應改為「最小相似度」
                # 重新計算一個近似的相似度值來與 SIMILARITY_THRESHOLD 比較
                
                # 更精確的處理方式是使用 Cosine Similarity (夾角餘弦)
                # 但 FAISS 預設是 L2 Distance。如果需要 Cosine Similarity，
                # Langchain 的 `from_documents` 在 FAISS 向量儲存預設就是用 L2 距離。
                # 如果要使用 Cosine Similarity，需要先正規化 embedding，
                # 或者在 `similarity_search_with_score` 時指定 `score_threshold` (但它也是基於距離的)。
                
                # 最簡單的方式：將 SIMILARITY_THRESHOLD 理解為「最大允許的距離」，而非相似度。
                # 或直接調整 SIMILARITY_THRESHOLD 的初始值，讓它代表距離。
                
                # 我將 SIMILARITY_THRESHOLD 變數名保持不變，但將其視為「所需的最低相似度」。
                # 由於 FAISS 返回的是 L2 距離 (distance)，距離越小越相似。
                # 我將距離反向轉換為一個簡單的相似度值，以便與 SIMILARITY_THRESHOLD 比較。
                # 這個轉換 `max(0, 1 - distance / some_max_expected_distance)` 更能表示相似度
                # 或者直接用 `score` 與 `SIMILARITY_THRESHOLD` 比較，但 SIMILARITY_THRESHOLD 需改為距離上限。
                
                # 為了簡化，我將 SIMILARITY_THRESHOLD 視為一個較高閾值，並讓 score (距離) 小於某個值才算相似。
                # 假設 SIMILARITY_THRESHOLD 0.75-0.9 代表較高相似度，對應 L2 距離就是較小的值。
                
                # 這裡保留原有的 1/(1+distance) 轉換，但強調這是近似。
                similarity = 1 / (1 + distance) 
                print(f"RAG 搜尋結果：L2 距離={distance:.4f}, 轉換後相似度={similarity:.4f}")

                # 判斷是否達標：轉換後相似度大於等於設定的閾值
                if similarity >= SIMILARITY_THRESHOLD:
                    print("RAG 相似度達標，使用內部資料回答。")
                    return doc.page_content
                else:
                    print("RAG 相似度未達標，返回預設訊息。")
                    return "抱歉，我只知道關於內部資料的資訊，無法回答您的問題。請嘗試提出更相關的問題。"

        except Exception as e:
            print(f"RAG 搜尋失敗: {e}")
            return "抱歉，在搜尋內部資料時發生錯誤，請稍後再試。"
    else:
        print("FAISS 向量索引未成功載入或建立。")
        return "抱歉，核心知識庫尚未準備好，請稍後再試。"

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
    gr.Markdown("# 內部知識庫問答機器人 (純檢索)")
    gr.Markdown("此機器人僅基於內部文件庫進行相似度檢索，不使用外部 LLM 生成回答。")
    with gr.Row():
        with gr.Column():
            qbox = gr.Textbox(label="請輸入問題")
            abox = gr.Textbox(label="回答", interactive=False) # 回答框不可編輯
            btn = gr.Button("送出")
            btn.click(fn=rag_answer, inputs=qbox, outputs=abox)

from gradio.routes import mount_gradio_app
app = mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    # 將 port 設定為 Render 預設的 10000
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
