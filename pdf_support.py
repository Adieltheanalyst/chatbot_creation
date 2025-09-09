from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import numpy as np
import ollama

def split_pages_into_chunks(text,max_length=300,overlap=50):
    chunks=[]
    start=0
    while start<len(text):
        end=start+max_length
        chunks.append(text[start:end])
        start=end-overlap
    return chunks

def extraction_from_pdf(path,max_length=300, overlap=50):

    content=""
    reader= PdfReader(path)
    all_chunks=[]
    for page_num,page in enumerate(reader.pages,start=0):
        text=page.extract_text()
        content += text
    page_chunks = split_pages_into_chunks(content.strip())
    all_chunks.extend(page_chunks)

    return all_chunks

chunks = extraction_from_pdf("coffe_shop_details.pdf", max_length=300, overlap=50)

model=SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./models")
embeddings= model.encode(chunks, convert_to_numpy=True)

np.save("embeddings.npy",embeddings)
with open("chunks.txt","w",encoding="utf-8") as f:
    for c in chunks:
        f.write(c + "\n")

def search(query,top_k=2):
    query_vec=model.encode([query], convert_to_numpy=True)
    query_vec=query_vec/np.linalg.norm(query_vec)
    doc_vecs=embeddings/np.linalg.norm(embeddings,axis=1,keepdims=True)
    scores = np.dot(doc_vecs,query_vec.T).flatten()
    top_ids=np.argsort(-scores)[:top_k]
    results=[(chunks[i], float(scores[i])) for i in top_ids]
    print("\nUser:",query)
    # for r in results:
    #     print("Bot (relevant info):", r[0])
    return results[:top_k]



def ask_ollama(retrived_chunks, user_question):
    messages = [
        {"role": "system", "content": """
        You are a friendly and conversational assistant.
        Use the provided context to answer the question naturally.
        Do not mention 'context' in your answer.
        If the answer is not in the context, simply say
        something like "Hmm, I'm not sure about that."
        """},
        {"role": "user", "content": f"""
        CONTEXT:
        {retrived_chunks}

        QUESTION:
        {user_question}
        """}
    ]

    response = ollama.chat(model="mistral", messages=messages)
    return response["message"]["content"]

question="What is the opening time on saturday?"
retrived_chunks=search(question)
print(ask_ollama(retrived_chunks, question))

