from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import numpy as np
import re


def split_text_into_chunks(text, max_length=300, overlap=50):
    sentences = re.split(r'(?<=[.!?]) +', text)  # split by sentence boundaries
    chunks = []
    current_chunk = []

    current_length = 0
    for sentence in sentences:
        if current_length + len(sentence) <= max_length:
            current_chunk.append(sentence)
            current_length += len(sentence)
        else:
            # finalize chunk
            chunks.append(" ".join(current_chunk))

            # add overlap: take last N chars of previous chunk
            if overlap > 0 and len(current_chunk) > 0:
                overlap_text = current_chunk[-1][-overlap:]
                current_chunk = [overlap_text, sentence]
                current_length = len(overlap_text) + len(sentence)
            else:
                current_chunk = [sentence]
                current_length = len(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def extraction_from_pdf(path,max_length=300, overlap=50):

    content=""
    reader= PdfReader(path)
    all_chunks=[]
    for page_num,page in enumerate(reader.pages,start=0):
        text=page.extract_text()
        if text:
            page_chunks=split_text_into_chunks(text.strip())
            all_chunks.extend(page_chunks)

    return all_chunks

chunks = extraction_from_pdf("coffee_shop.pdf", max_length=300, overlap=50)

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
    for r in results:
        print("Bot (relevant info):", r[0])
    return results[0]

print(search("Where is the coffee shop located?"))