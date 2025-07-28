# Adobe "Connecting the Dots" Hackathon - Round 1B Submission

This project is a solution for Round 1B, designed to act as an "intelligent document analyst." It extracts and ranks the most relevant sections from a collection of PDFs based on a specific persona and job description.

## Architecture Overview

The solution is built on a high-performance, two-stage "Retrieve & Rerank" pipeline, optimized to meet all hackathon constraints (CPU-only, <1GB model size, <60s execution, offline).

1.  **PDF Parsing**: A fast, heuristic-based parser using `PyMuPDF` extracts structured content (Title, H1, H2, H3) without relying on large models.
2.  **Model Optimization**: All neural models are converted to the `.onnx` format and subjected to `INT8` quantization. This reduces their size by ~4x and speeds up inference by ~2-3x.
3.  **Stage 1: Retrieval**: A quantized `all-MiniLM-L6-v2` bi-encoder generates vector embeddings to quickly find the top 100 relevant section candidates from all documents.
4.  **Stage 2: Reranking**: A quantized `ms-marco-MiniLM-L6-v2` cross-encoder performs a high-precision analysis on the candidates to produce the final importance ranking.
5.  **Sub-section Analysis**: `YAKE!`, a lightweight unsupervised keyword extractor, generates the "Refined Text" summary for the most relevant sections.

## How to Build and Run

### Prerequisites

* Docker installed and running.
* Python 3.9+ for the one-time setup script.

### Step 1: Prepare the Optimized Models (One-Time Setup)

Before building the Docker image, you must run the setup script to download the base models and convert them into the optimized ONNX format. This step requires an internet connection.

```bash
pip install -r requirements.txt

python setup.py