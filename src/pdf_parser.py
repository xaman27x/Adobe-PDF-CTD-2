import fitz  # PyMuPDF
from src.utils import clean_text

def extract_chunks_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks")
        for block in blocks:
            text = clean_text(block[4])
            if text:
                chunks.append({
                    "text": text,
                    "page": page_num,
                    "bbox": block[:4]
                })

    return chunks
