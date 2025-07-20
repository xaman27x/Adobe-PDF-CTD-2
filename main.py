import os
import json
import argparse
from src.pdf_parser import extract_chunks_from_pdf
from src.embedder import Embedder
from src.ranker import rank_chunks
from src.output_generator import build_output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True, help="Name of collection folder (e.g. Collection1)")
    args = parser.parse_args()

    collection_dir = os.path.join("input", args.collection)
    pdf_dir = os.path.join(collection_dir, "pdfs")

    # 🔍 Try both input file names
    input_json = os.path.join(collection_dir, "challenge1b_input.json")
    if not os.path.exists(input_json):
        input_json = os.path.join(collection_dir, "input.json")

    if not os.path.exists(input_json):
        raise FileNotFoundError(f"❌ Neither 'challenge1b_input.json' nor 'input.json' found in {collection_dir}")

    output_path = os.path.join("output", f"{args.collection}_output.json")

    with open(input_json, "r", encoding="utf-8") as f:
        input_meta = json.load(f)

    all_chunks = []
    for doc in input_meta["documents"]:
        file_path = os.path.join(pdf_dir, doc["filename"])
        chunks = extract_chunks_from_pdf(file_path)
        for chunk in chunks:
            chunk["document"] = doc["filename"]
        all_chunks.extend(chunks)

    embedder = Embedder()
    ranked = rank_chunks(
        input_meta["persona"]["role"],
        input_meta["job_to_be_done"]["task"],
        all_chunks,
        embedder
    )

    build_output(input_meta, ranked, output_path)
    print(f"✅ Output saved to {output_path}")

if __name__ == "__main__":
    main()
