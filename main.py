# main.py

import json
from pathlib import Path
from engine import HeliosEngine
import logging

COLLECTIONS_ROOT = Path("collections")
OUTPUT_ROOT = Path("outputs") # Store outputs in a separate root

def main():
    logging.info("Starting Adobe Hackathon Round 1B Submission - Helios Engine")
    OUTPUT_ROOT.mkdir(exist_ok=True)
    
    engine = HeliosEngine()

    collection_dirs = [d for d in COLLECTIONS_ROOT.iterdir() if d.is_dir()]
    if not collection_dirs:
        logging.warning(f"No collection directories found in '{COLLECTIONS_ROOT}'.")
        return

    for collection_path in collection_dirs:
        logging.info(f"--- Processing Collection: {collection_path.name} ---")
        input_file = collection_path / "input.json"
        pdf_dir = collection_path / "PDFs"
        
        # Create a matching output directory
        output_dir = OUTPUT_ROOT / collection_path.name
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "output.json"

        if not input_file.exists() or not pdf_dir.exists():
            logging.error(f"Skipping: Missing input.json or PDFs directory in {collection_path.name}.")
            continue
            
        try:
            with open(input_file, 'r', encoding='utf-8') as f: input_data = json.load(f)
            doc_paths = list(pdf_dir.glob("*.pdf"))
            persona = input_data.get("persona", {}).get("role", "User")
            job_to_be_done = input_data.get("job_to_be_done", {}).get("task", "Analyze.")

            analysis_result = engine.run_analysis(doc_paths, persona, job_to_be_done)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, ensure_ascii=False, indent=2)
            logging.info(f"Successfully wrote analysis to {output_file}")

        except Exception as e:
            logging.critical(f"Critical error in {collection_path.name}: {e}", exc_info=True)

if __name__ == "__main__":
    main()