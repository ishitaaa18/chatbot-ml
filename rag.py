from langchain.embeddings.base import Embeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
from langdetect import detect
import os
from typing import Optional

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


def ingest_pdf_for_colleges(
    file_path: str,
    college_id: str,
    db_dir: str = "vectorstores",
    k: int = 5,
    replace: bool = False
):
    """
    Ingest a PDF into a FAISS DB dedicated to a single college.

    * Stores each chunk with a unique doc_id so we can avoid duplicates
      and later delete by college or file.
    * Will create the FAISS index if it doesn't exist.
    * If `replace=True`, removes any previous chunks from this PDF first.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    pdf_name = os.path.basename(file_path)

    # Add metadata used for deduplication & deletion
    for idx, doc in enumerate(documents):
        doc.metadata.update({
            "source": pdf_name,
            "college_id": college_id,
            "doc_id": f"{college_id}:{pdf_name}:{idx}"
        })

    # Normalise + chunk
    documents = _normalize_docs(documents, file_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    db_path = os.path.join(db_dir, college_id, "faiss_index")
    index_file = os.path.join(db_path, "index.faiss")

    # Load existing index if it exists
    if os.path.isfile(index_file):
        db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

        # Remove existing chunks of this PDF if replace=True
        if replace:
            existing_ids = [
                doc_id
                for doc_id, doc in db.docstore._dict.items()
                if doc.metadata.get("source") == pdf_name
                and doc.metadata.get("college_id") == college_id
            ]
            if existing_ids:
                db.delete(ids=existing_ids)

        # Skip adding if all doc_ids already exist (prevents duplicates)
        existing_doc_ids = set(db.docstore._dict.keys())
        new_chunks = [c for c in chunks if c.metadata["doc_id"] not in existing_doc_ids]
        if new_chunks:
            db.add_documents(new_chunks)
    else:
        os.makedirs(db_path, exist_ok=True)
        db = FAISS.from_documents(chunks, embeddings)

    db.save_local(db_path)
    return db, db.as_retriever(search_kwargs={"k": k})


def delete_pdf_for_college(
    college_id: str,
    pdf_name: str,
    db_dir: str = "vectorstores"
) -> Optional[int]:
    """
    Delete every vector chunk belonging to a given PDF from a college's FAISS index.

    Returns the number of chunks removed, or None if no index exists.
    """
    db_path = os.path.join(db_dir, college_id, "faiss_index")
    index_file = os.path.join(db_path, "index.faiss")
    pdf_basename = os.path.basename(pdf_name)

    if not os.path.isfile(index_file):
        print(f"⚠️ No FAISS index found for college '{college_id}'.")
        return None

    db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

    ids_to_delete = [
        doc_id
        for doc_id, doc in db.docstore._dict.items()
        if doc.metadata.get("source") == pdf_basename
        and doc.metadata.get("college_id") == college_id
    ]

    if not ids_to_delete:
        print(f"ℹ️ No chunks found for {pdf_basename}")
        return 0

    # Use LangChain's delete wrapper (handles FAISS ID type conversion)
    db.delete(ids=ids_to_delete)
    db.save_local(db_path)

    print(f"✅ Deleted {len(ids_to_delete)} chunks from {pdf_basename}")
    return len(ids_to_delete)


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
