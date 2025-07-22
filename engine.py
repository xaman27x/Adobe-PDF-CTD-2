# engine.py

import fitz  # PyMuPDF
import numpy as np
import onnxruntime as ort
import json
import logging
import time
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from typing import List, Dict, Any
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

logging.basicConfig(level=logging.INFO, format='%(asctime)s - HELIOS - [%(levelname)s] - %(message)s')

class Engine:

    def __init__(self, model_path: Path = Path("models/quantized")):
        logging.info(f"Initializing Helios Engine from '{model_path}'...")
        snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", allow_patterns=["*.json", "vocab.txt", "tokenizer.json"], local_dir=model_path, local_dir_use_symlinks=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.session = ort.InferenceSession(model_path / "model_quantized.onnx")
        logging.info("Helios cognitive core is online.")

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Generates semantic embeddings using the ONNX model."""
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="np")
        outputs = self.session.run(None, dict(inputs))

        last_hidden_state = outputs[0]
        mask = inputs['attention_mask']
        masked_hidden_state = last_hidden_state * np.expand_dims(mask, axis=-1)
        summed_hidden_state = np.sum(masked_hidden_state, axis=1)
        summed_mask = np.sum(mask, axis=1, keepdims=True)
        embeddings = summed_hidden_state / summed_mask
        return embeddings

    def _extract_and_structure_text(self, doc_path: Path) -> List[Dict]:
        """Hybrid pass: Extracts text chunks and identifies headings structurally."""
        doc = fitz.open(doc_path)
        structured_content = []
        fontsizes = []
        for page in doc:
            for block in page.get_text("dict", flags=fitz.TEXTFLAGS_INHIBIT_SPACES)["blocks"]:
                if block['type'] == 0:
                     for line in block['lines']:
                         for span in line['spans']:
                             fontsizes.append(span['size'])
        
        if not fontsizes: return []
        body_size = max(set(fontsizes), key=fontsizes.count)

        for page_num, page in enumerate(doc, 1):
            blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_INHIBIT_SPACES)["blocks"]
            for block in blocks:
                if block['type'] == 0:
                    span = block['lines'][0]['spans'][0]
                    text = " ".join([s['text'] for l in block['lines'] for s in l['spans']]).strip()
                    is_heading = (span['size'] > body_size * 1.15) or ("bold" in span['font'].lower())
                    if len(text.split()) > 100: is_heading = False # Headings are short

                    structured_content.append({
                        "document": doc_path.name, "page_number": page_num,
                        "text": text, "is_heading": is_heading
                    })
        return structured_content

    def _get_sections_from_structure(self, structured_content: List[Dict]) -> List[Dict]:
        sections = []
        current_section = None
        for item in structured_content:
            if item['is_heading']:
                if current_section: sections.append(current_section)
                current_section = {
                    "document": item["document"], "page_number": item["page_number"],
                    "section_title": item["text"], "content": ""
                }
            elif current_section:
                current_section["content"] += item["text"] + "\n\n"
        if current_section: sections.append(current_section)
        return [sec for sec in sections if sec.get("content")]

    def _summarize_text(self, text, num_sentences=2):
        """Performs extractive summarization using TextRank."""
        sentences = [s.strip() for s in text.split('.') if len(s.split()) > 5]
        if len(sentences) <= num_sentences: return " ".join(sentences)

        sentence_embeddings = self._embed(sentences)
        sim_matrix = cosine_similarity(sentence_embeddings)
        
        nx_graph = nx.from_numpy_array(sim_matrix)
        scores = nx.pagerank(nx_graph)
        
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        summary = ". ".join([s for score, s in ranked_sentences[:num_sentences]])
        return summary + "."

    def run_analysis(self, doc_paths: List[Path], persona: str, job_to_be_done: str) -> Dict[str, Any]:
        
        semantic_query = f"As a {persona}, I need to {job_to_be_done}."
        query_embedding = self._embed([semantic_query])

        all_content = []
        for doc_path in tqdm(doc_paths, desc="Parsing Documents"):
            all_content.extend(self._extract_and_structure_text(doc_path))

        if not all_content: return {}

        sections = self._get_sections_from_structure(all_content)
        all_chunks = [item for item in all_content if not item['is_heading']]
        
        section_contents = [sec['content'] for sec in sections]
        section_embeddings = self._embed(section_contents)
        section_scores = cosine_similarity(query_embedding, section_embeddings)[0]

        ranked_sections = []
        for i, score in enumerate(section_scores): sections[i]['score'] = score
        sections.sort(key=lambda x: x['score'], reverse=True)
        
        for rank, sec in enumerate(sections[:10], 1):
            ranked_sections.append({
                "document": sec["document"], "section_title": sec["section_title"],
                "importance_rank": rank, "page_number": sec["page_number"]
            })

        chunk_texts = [chunk['text'] for chunk in all_chunks]
        chunk_embeddings = self._embed(chunk_texts)
        chunk_scores = cosine_similarity(query_embedding, chunk_embeddings)[0]

        subsection_analysis = []
        for i, score in enumerate(chunk_scores): all_chunks[i]['score'] = score
        all_chunks.sort(key=lambda x: x['score'], reverse=True)

        for chunk in all_chunks[:5]:
             summary = self._summarize_text(chunk["text"])
             subsection_analysis.append({
                "document": chunk["document"], "refined_text": summary,
                "page_number": chunk["page_number"]
            })

        output = {
            "metadata": {
                "input_documents": [p.name for p in doc_paths], "persona": persona,
                "job_to_be_done": job_to_be_done, "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "extracted_sections": ranked_sections,
            "subsection_analysis": subsection_analysis
        }
        
        return output