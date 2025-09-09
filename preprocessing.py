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
        content += text
    page_chunks = split_pages_into_chunks(content.strip())
    all_chunks.extend(page_chunks)

    return all_chunks

chunks = extraction_from_pdf("coffe_shop_details.pdf", max_length=300, overlap=50)

model=SentenceTransformer("multi-qa-MiniLM-L6-cos-v1", cache_folder="./models")
embeddings= model.encode(chunks, convert_to_numpy=True)

np.save("embeddings.npy",embeddings)
with open("chunks.txt","w",encoding="utf-8") as f:
    for c in chunks:
        f.write(c + "\n")
print("✅ Preprocessing complete. Saved embeddings.npy and chunks.txt")
