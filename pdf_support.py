import numpy as np
import ollama
import streamlit as st
from transformers import pipeline, AutoTokenizer,AutoModel
embeddings=np.load("embeddings.npy")

with open("chunks.txt", "r", encoding="utf-8") as f:
    chunks = [line.strip() for line in f if line.strip()]

from sentence_transformers import SentenceTransformer


def search(query,top_k=5):
    local_model_path=r"C:\Users\gacha\PycharmProjects\chatbot\models\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    model=SentenceTransformer(local_model_path)
    query_vec = model.encode([query], convert_to_numpy=True)
    query_vec = query_vec/np.linalg.norm(query_vec)
    doc_vecs = embeddings/np.linalg.norm(embeddings,axis=1,keepdims=True)
    scores = np.dot(doc_vecs,query_vec.T).flatten()
    top_ids = np.argsort(-scores)[:top_k]
    results = [(chunks[i], float(scores[i])) for i in top_ids]
    return results[:top_k]



def ask_ollama(retrived_chunks, user_question):
    messages = [
        {"role": "system", "content": """
        You are a friendly,conversational and official or Adiel's coffee Corner.
        Always speak in the first person plural ("we", "our") as if you are representing the café directly.
        Keep answers friendly, concise, and welcoming.
        Use the provided context to answer the question naturally.
        Do not mention 'context' in your answer.
        If the answer is not in the context, simply say
        something like "Hmm, I'm not sure about that."
        """},
        {"role": "user", "content": f"""
        CONTEXT:
        {retrived_chunks}\n\nQUESTION:{user_question}
        """}
    ]

    response = ollama.chat(model="phi:2.7b", messages=messages)
    return response["message"]["content"]


# while True:
#     question=input("\nYou: ")
#     if question .lower() in ["quit","exit"]:
#         break
#     retrived_chunks = search(question)
#     answer= ask_ollama(retrived_chunks,question)
#     print("Bot:", answer)
bot_name="Natalie"


if "messages" not in st.session_state:
    st.session_state.messages=[
        {"role": "bot", "content": f"Hi there 👋,I'm {bot_name} How can I help you today "}
    ]
    st.session_state.chat_active=True
st.title("Adiel's Coffee Corner Chatbot ☕")

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You** {msg['content']}")
    else:
        st.markdown(f"**{bot_name}:** {msg['content']}")

def send_message():
    question=st.session_state.question_input.strip()
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    if question.lower() in ["quit","exit"]:
        st.session_state.messages.append(
            {"role": "bot", "content": f"Goodbye 👋 Thanks for visiting Adiel's Coffee Corner! ☕"}
        )
        st.session_state.chat_active = False
    else:
        st.session_state.messages.append({"role":"user","content":question})
        retrived_chunks = search(question)
        answer= ask_ollama(retrived_chunks,question)
        st.session_state.messages.append({"role":"bot","content": answer})
    st.session_state.question_input=""

# question= st.text_input("Talk to us: ",key="question_input",on_change=send_message)
if st.session_state.chat_active:
    st.text_input("Talk to us:", key="question_input", on_change=send_message)
else:
    st.info("🔒 Chat ended. Refresh the page to start a new conversation.")