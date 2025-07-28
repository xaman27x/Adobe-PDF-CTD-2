import fitz
import statistics
from collections import defaultdict

def get_doc_styles(doc):
    """Identifies the most common font style (name, size) to define body text."""
    font_counts = defaultdict(int)
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        font_counts[(span["size"], span["font"])] += 1
    
    if not font_counts:
        return 12.0, "default"
        
    modal_style = max(font_counts, key=font_counts.get)
    return modal_style[0], modal_style[1]

def parse_pdf_sections(pdf_path: str):
    """
    Extracts structured sections (Title, H1, H2, H3) from a PDF document.

    Yields:
        dict: A dictionary for each identified section.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening {pdf_path}: {e}")
        return

    if not doc or doc.page_count == 0:
        return

    body_size, body_font = get_doc_styles(doc)
    
    # Title Heuristic: Largest font on the first page
    title_text = ""
    max_font_size = 0
    first_page_blocks = doc[0].get_text("dict")["blocks"]
    for block in first_page_blocks:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["size"] > max_font_size:
                        max_font_size = span["size"]
                        title_text = line["spans"][0]["text"].strip()
    
    if title_text and max_font_size > body_size * 1.2:
        yield {
            "level": "Title",
            "text": title_text,
            "page": 1,
            "doc_name": os.path.basename(pdf_path)
        }

    font_sizes = defaultdict(list)
    for page_num, page in enumerate(doc, 1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    span = line["spans"][0]
                    is_bold = "bold" in span["font"].lower()
                    is_larger = span["size"] > body_size * 1.05
                    is_short = len(span["text"]) < 150 
                    
                    if (is_larger or is_bold) and is_short and not span["text"].strip().endswith('.'):
                        font_sizes[span["size"]].append({
                            "text": span["text"].strip(),
                            "page": page_num
                        })

    if not font_sizes:
        return

    unique_sizes = sorted(font_sizes.keys(), reverse=True)
    level_map = {
        0: "H1",
        1: "H2",
        2: "H3"
    }

    for i, size in enumerate(unique_sizes):
        if i > 2:
            break
        level = level_map[i]
        for heading_info in font_sizes[size]:
            yield {
                "level": level,
                "text": heading_info["text"],
                "page": heading_info["page"],
                "doc_name": os.path.basename(pdf_path)
            }