from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.llms import Ollama
from deep_translator import GoogleTranslator
from rag import auto_ingest_data_folder
import os

# -----------------------------
# Setup Memory & LLM
# -----------------------------
session_id = "user1-session"

redis_history = RedisChatMessageHistory(
    url="redis://localhost:6379/0",
    session_id=session_id
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    output_key="answer",
    chat_memory=redis_history,
    return_messages=True
)

# Ollama LLM (Mistral 7B running locally)
llm = Ollama(
    model="mistral",
    base_url="http://127.0.0.1:11434"
)

# Resolve absolute data path (always works regardless of where you run python)
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "data")

retriever = auto_ingest_data_folder(data_path)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    output_key="answer"
)

# -----------------------------
# Chat Loop
# -----------------------------
print("🌍 Multilingual Chatbot Ready! Type 'quit' to exit.\n")

while True:
    query = input("You: ")
    if query.strip().lower() in ["quit", "exit", "bye"]:
        print("👋 Goodbye!")
        break

    try:
        # Detect source language
        detected_lang = GoogleTranslator(source="auto", target="en").translate(query)
        query_en = detected_lang  # query translated to English

        # Run retrieval + LLM answer
        result = qa_chain.invoke({"question": query_en})
        answer_en = result["answer"]

        # Retrieved docs
        retrieved_docs = result.get("source_documents", [])

        # Fallback: if no docs OR empty answer
        if not retrieved_docs or not answer_en.strip():
            fallback_msg = "This information is not in the official documents."
            answer_final = GoogleTranslator(source="en", target="auto").translate(fallback_msg)
            print(f"🤖 {answer_final}")
            continue

        # Translate final answer back into original query’s language
        answer_final = GoogleTranslator(source="en", target="auto").translate(answer_en)

        sources = [doc.metadata.get("source", "unknown") for doc in retrieved_docs]

        print(f"🤖 {answer_final}")
        if sources:
            print(f"📖 Sources: {set(sources)}")

    except Exception as e:
        print(f"⚠️ Error: {e}")
