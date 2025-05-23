import os
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader


def load_documents_from_folder(folder_path: str) -> List[Document]:
    """
    遍歷資料夾，僅載入 .txt 檔案並回傳 Document 列表。
    """
    docs: List[Document] = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            try:
                loader = TextLoader(filepath, autodetect_encoding=True)
                docs.extend(loader.load())
            except Exception as e:
                print(f"讀取失敗 {filename}: {e}")
    return docs
