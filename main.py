import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import gradio as gr
from typing import List, Tuple

# 改用開源嵌入模型
# 使用新的導入路徑以避免 LangChainDeprecationWarning
from langchain_huggingface import HuggingFaceEmbeddings # <--- IMPORTANT CHANGE HERE

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from utils import load_documents_from_folder # 確保 utils.py 檔案在同一個目錄下

# LINE Bot SDK
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

# ====== 環境變數設定 (只保留 LINE Bot 相關，移除 LLM API Key) ======
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
# 調整相似度閾值，0.75-0.85 較常見，可依需求調整
# L2 距離越小越相似，這裡的 SIMILARITY_THRESHOLD 是轉換後的相似度
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))

# ====== 模型與向量索引設定 ======
# 使用您找到的 "tiny" 模型，例如 'cointegrated/rubert-tiny' 或 'sergeyzh/rubert-tiny-sts'
# 為了確保模型能夠處理英文或其他常用語言，我建議嘗試以下其中一個：
# embedding_model_name = "cointegrated/rubert-tiny" # <--- 您在圖中找到的一個
# embedding_model_name = "sergeyzh/rubert-tiny-sts" # <--- 您在圖中找到的另一個
# embedding_model_name = "johnpaulbin/bge-m3-distilled-tiny" # 另一個可能更通用的英文小模型

# 這裡先選用一個，如果不行再換另一個 "tiny" 模型
# 根據您的截圖，我會選用 cointegrated/rubert-tiny 或 sergeyzh/rubert-tiny-sts，它們是 sentence-transformers 系列的。
# 選擇一個實際存在且能用於 sentence_transformers 的 tiny 模型。
# 這裡假設 'cointegrated/rubert-tiny' 是一個好的起點。
embedding_model_name = "cointegrated/rubert-tiny" # <--- 將此行替換為您選擇的 tiny 模型

embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

VECTOR_STORE_PATH = "./faiss_index"
DOCUMENTS_PATH = "./docs" # 這個在部署時其實可以不用讀取，但為了防止意外，保留路徑定義

# 初始化 LINE Bot
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# 確保資料目錄與索引資料夾
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(DOCUMENTS_PATH, exist_ok=True)
vectorstore: FAISS = None # 全局變數用於儲存載入的向量儲存

# ====== 向量索引建立 (這個函數主要用於本地建立索引，部署時應避免執行) ======
def build_vector_store() -> FAISS:
    docs = load_documents_from_folder(DOCUMENTS_PATH)
    if not docs:
        # 在部署環境下，如果沒有預先建立索引且 docs 為空，這裡會出錯
        # 這是期望的行為，因為應該要有預先建立的索引
        raise RuntimeError(f"在 '{DOCUMENTS_PATH}' 資料夾中找不到文件，無法建立索引。請確保已預先建立 FAISS 索引。")
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = splitter.split_documents(docs)
    faiss_db = FAISS.from_documents(texts, embedding_model)
    faiss_db.save_local(VECTOR_STORE_PATH)
    return faiss_db

# 載入或建立索引 (部署時主要執行載入，移除重建回退邏輯以避免 OOM)
def ensure_vectorstore():
    global vectorstore
    if vectorstore is not None: # 如果已經載入過，就直接返回
        return

    # 檢查 faiss_index 資料夾是否存在 index.faiss 和 index.pkl
    faiss_index_exists = os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")) and \
                         os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.pkl"))

    if not faiss_index_exists:
        # 在部署環境下，如果沒有預先建立索引，這將是一個致命錯誤
        print("致命錯誤：未找到 FAISS 索引。請確認已在本地建立索引並將 'faiss_index' 資料夾推送到 Git！")
        vectorstore = None # 設置為 None，讓 rag_answer 返回錯誤訊息
        # 您甚至可以選擇在這裡拋出異常，讓服務直接啟動失敗，而不是繼續運行一個沒有知識庫的服務
        # raise RuntimeError("FAISS 索引缺失，應用程式無法啟動。")
    else:
        print("載入現有 FAISS 索引...")
        try:
            vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embedding_model)
            print("FAISS 索引載入完成。")
        except Exception as e:
            print(f"致命錯誤：載入 FAISS 索引失敗: {e}")
            print("請檢查 'faiss_index' 資料夾的完整性，並確認 'embedding_model_name' 是否與建立索引時一致。")
            vectorstore = None # 設置為 None，讓 rag_answer 返回錯誤訊息

# ====== FAISS RAG (純檢索) 問答邏輯 ======
def rag_answer(question: str) -> str:
    # 確保 vectorstore 在處理請求前被載入
    ensure_vectorstore()

    if vectorstore:
        try:
            docs_and_scores: List[Tuple] = vectorstore.similarity_search_with_score(question, k=1)
            
            if docs_and_scores:
                doc, score = docs_and_scores[0]
                distance = score
                # 這裡的 SIMILARITY_THRESHOLD 是針對轉換後的相似度
                # L2 距離越小越好，我們將其反向映射到一個 0-1 相似度
                # 簡單轉換：相似度 = 1 / (1 + 距離)
                # 距離為 0 時相似度為 1；距離越大，相似度越趨近 0
                similarity = 1 / (1 + distance)
                print(f"RAG 搜尋結果：L2 距離={distance:.4f}, 轉換後相似度={similarity:.4f}")

                if similarity >= SIMILARITY_THRESHOLD:
                    print("RAG 相似度達標，使用內部資料回答。")
                    return doc.page_content
                else:
                    print("RAG 相似度未達標，返回預設訊息。")
                    return "抱歉，我只知道關於內部資料的資訊，無法回答您的問題。請嘗試提出更相關的問題。"
            else:
                print("未找到任何相關文檔。")
                return "抱歉，我只知道關於內部資料的資訊，無法回答您的問題。請嘗試提出更相關的問題。"

        except Exception as e:
            print(f"RAG 搜尋失敗: {e}")
            return "抱歉，在搜尋內部資料時發生錯誤，請稍後再試。"
    else:
        print("FAISS 向量索引未成功載入。無法進行檢索。")
        return "抱歉，核心知識庫尚未準備好，請聯繫管理員。"

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
    
    # 在啟動 Uvicorn 服務之前，確保向量索引已經載入
    ensure_vectorstore() 

    # 只有當 vectorstore 成功載入時才啟動服務
    if vectorstore is not None:
        uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
    else:
        print("由於 FAISS 索引載入失敗，應用程式無法正常啟動。")
        exit(1) # 強制退出，避免在沒有知識庫的情況下運行
