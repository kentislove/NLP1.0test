import os
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader
)

def load_documents_from_folder(folder_path: str) -> List[Document]:
    docs = []
    for file in os.listdir(folder_path):
        filepath = os.path.join(folder_path, file)
        if file.endswith(".txt"):
            loader = TextLoader(filepath, autodetect_encoding=True)
        elif file.endswith(".pdf"):
            loader = UnstructuredPDFLoader(filepath)
        elif file.endswith(".doc") or file.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(filepath)
        elif file.endswith(".xlsx") or file.endswith(".xls"):
            loader = UnstructuredExcelLoader(filepath)
        elif file.endswith(".csv"):
            docs.extend(parse_csv_file(filepath))
            continue
        else:
            print(f"不支援的格式：{file}")
            continue
        try:
            docs.extend(loader.load())
        except Exception as e:
            print(f"讀取失敗 {file}: {e}")
    return docs

def parse_csv_file(filepath: str) -> List[Document]:
    import csv
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        for row in reader:
            combined = "\n".join(f"{h}: {r}" for h, r in zip(header, row))
            rows.append(Document(page_content=combined, metadata={"source": filepath}))
    return rows
