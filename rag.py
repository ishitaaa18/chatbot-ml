from langchain.embeddings.base import Embeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
from langdetect import detect
import os

# -----------------------------
# Embeddings setup (Multilingual LaBSE)
# -----------------------------
embedding_model = SentenceTransformer("sentence-transformers/LaBSE")

class LocalEmbeddings(Embeddings):
    """Wrapper to make SentenceTransformer compatible with LangChain FAISS."""

    def embed_documents(self, texts):
        return embedding_model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return embedding_model.encode([text], convert_to_numpy=True).tolist()[0]

# ✅ Instantiate once
embeddings = LocalEmbeddings()

# -----------------------------
# Helpers
# -----------------------------
def _normalize_docs(documents, file_path):
    """
    Ensure all docs are in English:
    - Auto-detect language
    - Translate to English if not already English
    """
    translator = GoogleTranslator(source="auto", target="en")
    for doc in documents:
        try:
            lang = detect(doc.page_content[:200])  # detect based on snippet
            if lang != "en":
                translated_text = translator.translate(doc.page_content)
                doc.page_content = translated_text
        except Exception as e:
            print(f"⚠️ Normalization failed for {file_path}: {e}")
    return documents


def ingest_text(file_path):
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    for doc in documents:
        doc.metadata["source"] = os.path.basename(file_path)

    documents = _normalize_docs(documents, file_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    return FAISS.from_documents(chunks, embeddings)


def ingest_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    for doc in documents:
        doc.metadata["source"] = os.path.basename(file_path)

    documents = _normalize_docs(documents, file_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    return FAISS.from_documents(chunks, embeddings)


def ingest_pdf_for_colleges(file_path, college_id, db_dir="vectorstores"):
    """Ingest college PDFs into persistent FAISS DB (per college)."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    for doc in documents:
        doc.metadata["source"] = os.path.basename(file_path)

    documents = _normalize_docs(documents, file_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, f"{college_id}_faiss")

    if os.path.exists(db_path):
        db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(chunks)
    else:
        db = FAISS.from_documents(chunks, embeddings)

    db.save_local(db_path)
    return db.as_retriever(search_kwargs={"k": 5})


# -----------------------------
# Auto-ingest Folder
# -----------------------------
def auto_ingest_data_folder(folder_path="data"):
    """Ingest all PDFs/TXT files in a folder into one FAISS retriever."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    abs_folder_path = os.path.join(current_dir, folder_path)

    if not os.path.exists(abs_folder_path):
        raise ValueError(f"❌ Folder not found: {abs_folder_path}")

    files = [f for f in os.listdir(abs_folder_path) if f.endswith((".txt", ".pdf"))]
    if not files:
        raise ValueError("❌ No valid .txt or .pdf files found in data folder")

    print(f"📂 Found raw files in {os.path.abspath(abs_folder_path)}: {files}")

    vectorstore = None
    for fname in files:
        fpath = os.path.join(abs_folder_path, fname)
        if fname.endswith(".txt"):
            print(f"➡️ Ingesting TXT: {fname}")
            vs = ingest_text(fpath)
        elif fname.endswith(".pdf"):
            print(f"➡️ Ingesting PDF: {fname}")
            vs = ingest_pdf(fpath)
        else:
            continue

        if vectorstore is None:
            vectorstore = vs
        else:
            vectorstore.merge_from(vs)

    print(f"✅ Ingested {len(files)} files: {files}")
    return vectorstore.as_retriever(search_kwargs={"k": 5})
