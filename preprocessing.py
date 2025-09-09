import fitz  # PyMuPDF

pdf_path = r"C:\Users\gacha\PycharmProjects\chatbot\coffe_shop_details.pdf"

# Open the PDF
doc = fitz.open(pdf_path)

# Extract all text at once
full_text = doc.get_toc("text")  # This gets the whole text of the PDF in reading order

# Split into paragraphs
# Assuming paragraphs are separated by at least one blank line
paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip()]

# Print example paragraphs
for i, para in enumerate(paragraphs):
    print(f"--- Paragraph {i+1} ---")
    print(para)
    print()
