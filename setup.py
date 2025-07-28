# setup.py
import os
from pathlib import Path
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTModelForSequenceClassification
from onnxruntime.quantization import quantize_dynamic, QuantType

def setup_models():
    """
    Downloads, converts, and quantizes the necessary transformer models.
    This is a one-time setup step to be run before building the Docker image.
    """
    BI_ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    bi_encoder_path = models_dir / "bi_encoder"
    cross_encoder_path = models_dir / "cross_encoder"
    
    quantized_bi_encoder_path = models_dir / "bi_encoder_quantized.onnx"
    quantized_cross_encoder_path = models_dir / "cross_encoder_quantized.onnx"

    # --- Part 1: Process Bi-Encoder (Sentence Transformer) ---
    if not quantized_bi_encoder_path.exists():
        print(f"\n[1/4] Downloading Bi-Encoder: {BI_ENCODER_NAME}")
        # We download the model using sentence-transformers to get the correct pooling layer setup
        sbert_model = SentenceTransformer(BI_ENCODER_NAME)
        sbert_model.save(str(bi_encoder_path))
        
        print(f"[2/4] Converting Bi-Encoder to ONNX...")
        # Use Optimum to export to ONNX format for feature extraction
        onnx_bi_encoder = ORTModelForFeatureExtraction.from_pretrained(str(bi_encoder_path), export=True)
        onnx_bi_encoder.save_pretrained(bi_encoder_path)
        
        # Now quantize the exported model
        print(f"[3/4] Quantizing Bi-Encoder ONNX model...")
        unquantized_model_path = bi_encoder_path / "model.onnx"
        quantize_dynamic(
            model_input=unquantized_model_path,
            model_output=quantized_bi_encoder_path,
            weight_type=QuantType.QInt8
        )
        print(f"   -> Saved quantized Bi-Encoder to {quantized_bi_encoder_path}")
    else:
        print(f"\n[OK] Quantized Bi-Encoder already exists at {quantized_bi_encoder_path}")


    # --- Part 2: Process Cross-Encoder (Reranker) ---
    if not quantized_cross_encoder_path.exists():
        print(f"\n[1/3] Downloading Cross-Encoder: {CROSS_ENCODER_NAME}")
        # We only need the tokenizer and model for conversion
        tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER_NAME)
        tokenizer.save_pretrained(str(cross_encoder_path))

        print(f"[2/3] Converting Cross-Encoder to ONNX...")
        # Use Optimum for sequence classification export
        onnx_cross_encoder = ORTModelForSequenceClassification.from_pretrained(CROSS_ENCODER_NAME, export=True)
        onnx_cross_encoder.save_pretrained(cross_encoder_path)
        
        print(f"[3/3] Quantizing Cross-Encoder ONNX model...")
        unquantized_model_path = cross_encoder_path / "model.onnx"
        quantize_dynamic(
            model_input=unquantized_model_path,
            model_output=quantized_cross_encoder_path,
            weight_type=QuantType.QInt8
        )
        print(f"   -> Saved quantized Cross-Encoder to {quantized_cross_encoder_path}")
    else:
         print(f"\n[OK] Quantized Cross-Encoder already exists at {quantized_cross_encoder_path}")

    print("\n--- Model Setup Complete ---")
    print(f"Optimized models are ready in the '{models_dir}' directory.")


if __name__ == "__main__":
    setup_models()