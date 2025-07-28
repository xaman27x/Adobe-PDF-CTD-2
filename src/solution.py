# src/solution.py
import os
import json
import time
import numpy as np
import onnxruntime as ort
from pathlib import Path
from transformers import AutoTokenizer
import yake
from .parser import parse_pdf_sections

INPUT_DIR = Path("/app/input")
OUTPUT_DIR = Path("/app/output")
MODELS_DIR = Path("models")

BI_ENCODER_ONNX_PATH = MODELS_DIR / "bi_encoder_quantized.onnx"
BI_ENCODER_TOKENIZER_PATH = MODELS_DIR / "bi_encoder"
CROSS_ENCODER_ONNX_PATH = MODELS_DIR / "cross_encoder_quantized.onnx"
CROSS_ENCODER_TOKENIZER_PATH = MODELS_DIR / "cross_encoder"

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(float)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.maximum(np.sum(input_mask_expanded, axis=1), 1e-9)
    return sum_embeddings / sum_mask

def cosine_similarity(a, b):
    return np.dot(a, b.T) / (np.linalg.norm(a, axis=1)[:, np.newaxis] * np.linalg.norm(b, axis=1))

# --- Main Classes ---
class BiEncoder:
    def __init__(self):
        print("Loading Bi-Encoder (Sentence Transformer)...")
        self.session = ort.InferenceSession(str(BI_ENCODER_ONNX_PATH))
        self.tokenizer = AutoTokenizer.from_pretrained(str(BI_ENCODER_TOKENIZER_PATH))
    
    def encode(self, texts, batch_size=32):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encoded_input = self.tokenizer(batch, padding=True, truncation=True, return_tensors='np', max_length=512)
            model_output = self.session.run(None, dict(encoded_input))
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            all_embeddings.append(sentence_embeddings)
        return np.vstack(all_embeddings)

class CrossEncoder:
    def __init__(self):
        print("Loading Cross-Encoder (Reranker)...")
        self.session = ort.InferenceSession(str(CROSS_ENCODER_ONNX_PATH))
        self.tokenizer = AutoTokenizer.from_pretrained(str(CROSS_ENCODER_TOKENIZER_PATH))

    def predict(self, pairs, batch_size=32):
        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            features = self.tokenizer(batch, padding=True, truncation=True, return_tensors="np", max_length=512)
            outputs = self.session.run(None, dict(features))
            scores = 1 / (1 + np.exp(-outputs[0])) # Apply sigmoid to logits
            all_scores.extend(scores.flatten())
        return np.array(all_scores)

class DocumentAnalyst:
    def __init__(self):
        self.bi_encoder = BiEncoder()
        self.cross_encoder = CrossEncoder()
        self.keyphrase_extractor = yake.KeywordExtractor(n=3, dedupLim=0.9, top=5)

    def run(self):
        start_time = time.time()
        
        pdf_files = list(INPUT_DIR.glob("*.pdf"))
        with open(INPUT_DIR / "persona.txt", "r") as f:
            persona = f.read()
        with open(INPUT_DIR / "job.txt", "r") as f:
            job = f.read()
        
        query = f"Persona: {persona}\nJob: {job}"
        
        print("Parsing PDFs and extracting sections...")
        all_sections = []
        for pdf_path in pdf_files:
            sections = list(parse_pdf_sections(str(pdf_path)))
            all_sections.extend(sections)
        
        if not all_sections:
            print("No sections found in any PDF. Exiting.")
            return

        section_texts = [s['text'] for s in all_sections]

        print("Stage 1: Retrieving candidates with Bi-Encoder...")
        query_embedding = self.bi_encoder.encode([query])
        section_embeddings = self.bi_encoder.encode(section_texts)
        
        similarities = cosine_similarity(query_embedding, section_embeddings)[0]
        
        top_k = min(100, len(similarities))
        candidate_indices = np.argsort(similarities)[-top_k:][::-1]

        print("Stage 2: Reranking candidates with Cross-Encoder...")
        rerank_pairs = [[query, section_texts[i]] for i in candidate_indices]
        rerank_scores = self.cross_encoder.predict(rerank_pairs)

        reranked_results = []
        for i, score in zip(candidate_indices, rerank_scores):
            result = all_sections[i]
            result['score'] = score
            reranked_results.append(result)
            
        # Sort by the new cross-encoder score
        reranked_results.sort(key=lambda x: x['score'], reverse=True)

        # 5. Sub-section Analysis & Output Formatting
        print("Finalizing output...")
        output_json = {
            "metadata": {
                "input_documents": [p.name for p in pdf_files],
                "persona": persona,
                "job_to_be_done": job,
                "processing_timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            },
            "extracted_sections": [],
            "sub_section_analysis": []
        }

        for rank, result in enumerate(reranked_results, 1):
            output_json["extracted_sections"].append({
                "document": result['doc_name'],
                "page_number": result['page'],
                "section_title": result['text'],
                "importance_rank": rank
            })
            
            keywords = self.keyphrase_extractor.extract_keywords(result['text'])
            refined_text = ", ".join([kw[0] for kw in keywords])

            output_json["sub_section_analysis"].append({
                "document": result['doc_name'],
                "page_number": result['page'],
                "refined_text": refined_text
            })

        output_path = OUTPUT_DIR / "challenge1b_output.json"
        with open(output_path, "w") as f:
            json.dump(output_json, f, indent=2)

        end_time = time.time()
        print(f"Processing complete in {end_time - start_time:.2f} seconds.")
        print(f"Output written to {output_path}")

if __name__ == "__main__":
    analyst = DocumentAnalyst()
    analyst.run()