from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.llms import Ollama
from extract import auto_ingest_data_folder

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
    input_key="question",   # match the chain input
    output_key="answer",    # match the chain output
    chat_memory=redis_history,
    return_messages=True
)

# Ollama LLM (Mistral 7B running locally)
llm = Ollama(model="mistral",
             base_url="http://127.0.0.1:11435")

# Ingest documents
retriever = auto_ingest_data_folder("data")

# Setup Conversational RAG chain
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
        result = qa_chain.invoke({"question": query})
        answer = result["answer"]
        sources = [doc.metadata.get("source", "unknown") for doc in result["source_documents"]]

        print(f"🤖 {answer}")
        if sources:
            print(f"📖 Sources: {set(sources)}")

        # 🔍 Debug: show retrieved snippets (first 200 chars each)
        # for doc in result["source_documents"]:
        #     snippet = doc.page_content[:200].replace("\n", " ")
        #     print(f"📄 Snippet from {doc.metadata.get('source','?')}: {snippet}...")

    except Exception as e:
        print(f"⚠️ Error: {e}")
