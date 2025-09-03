# multilingual_chatbot.py

from googletrans import Translator
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Initialize translator + memory
translator = Translator()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def build_chatbot(retriever, llm):
    """
    Build a conversational retrieval chatbot with memory.
    retriever: FAISS retriever (from rag.py)
    llm: language model (OpenAI, Mistral via Ollama, etc.)
    """
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return qa

def chatbot_query(query, qa):
    """
    Handles multilingual queries:
    - Detect language
    - Translate to English
    - Query retriever + LLM
    - Translate answer back to original language
    """
    # detect + translate to English
    detected_lang = translator.detect(query).lang
    query_en = translator.translate(query, src=detected_lang, dest="en").text
    
    # run retrieval + generation
    answer_en = qa({"question": query_en})["answer"]
    
    # translate back if needed
    if detected_lang != "en":
        answer = translator.translate(answer_en, src="en", dest=detected_lang).text
    else:
        answer = answer_en
    return answer

# Example usage (run only if this file is executed directly)
if __name__ == "__main__":
    from rag import retriever, llm   # Import your teammate's retriever + llm
    
    qa = build_chatbot(retriever, llm)
    
    print("Multilingual Chatbot ready! Type 'quit' to exit.")
    while True:
        query = input("You: ")
        if query.lower() in ["quit", "exit"]:
            break
        response = chatbot_query(query, qa)
        print("Bot:", response)
