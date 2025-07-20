import os
import json
from datetime import datetime
from src.utils import clean_text

def build_output(input_meta, ranked_chunks, output_path):
    metadata = {
        "input_documents": [doc["filename"] for doc in input_meta["documents"]],
        "persona": input_meta["persona"]["role"],
        "job_to_be_done": input_meta["job_to_be_done"]["task"],
        "processing_timestamp": datetime.now().isoformat()
    }

    extracted_sections = []
    subsection_analysis = []

    for chunk in ranked_chunks:
        extracted_sections.append({
            "document": chunk["document"],
            "section_title": chunk["text"][:80] + ("..." if len(chunk["text"]) > 80 else ""),
            "importance_rank": chunk["importance_rank"],
            "page_number": chunk["page"]
        })

        subsection_analysis.append({
            "document": chunk["document"],
            "refined_text": clean_text(chunk["text"]),
            "page_number": chunk["page"]
        })

    output = {
        "metadata": metadata,
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)
