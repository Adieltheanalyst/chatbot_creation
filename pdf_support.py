from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import numpy as np


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
        if text:
            page_chunks=split_pages_into_chunks(text.strip())
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
    return print(results[0])

print(search("What time does is the coffe shop open on sunday?"))