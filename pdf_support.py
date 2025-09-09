import numpy as np
import ollama

embeddings=np.load("embeddings.npy")

with open("chunks.txt", "r", encoding="utf-8") as f:
    chunks = [line.strip() for line in f if line.strip()]

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./models")

def search(query,top_k=5):
    query_vec=model.encode([query], convert_to_numpy=True)
    query_vec=query_vec/np.linalg.norm(query_vec)
    doc_vecs=embeddings/np.linalg.norm(embeddings,axis=1,keepdims=True)
    scores = np.dot(doc_vecs,query_vec.T).flatten()
    top_ids=np.argsort(-scores)[:top_k]
    results=[(chunks[i], float(scores[i])) for i in top_ids]
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


while True:
    question=input("\nYou: ")
    if question .lower() in ["quit","exit"]:
        break
    retrived_chunks = search(question)
    answer= ask_ollama(retrived_chunks,question)
    print("Bot:", answer)

