# setup_model.py

from pathlib import Path
from optimum.onnxruntime import ORTQuantizer, ORTModel
from optimum.onnxruntime.configuration import AutoQuantizationConfig

def setup_model():
    """
    This script should be run once before building the final Docker image.
    """
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    onnx_path = Path("models/onnx")
    quantized_path = Path("models/quantized")
    

    # Step 1
    model = ORTModel.from_pretrained(model_id, export=True)
    model.save_pretrained(onnx_path)
    print("ONNX export complete.")

    # Step 2
    qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)

    # Step 3
    print(f"Step 3: Quantizing the model and saving to {quantized_path}...")
    quantizer = ORTQuantizer.from_pretrained(onnx_path)
    quantizer.quantize(save_dir=quantized_path, quantization_config=qconfig)


if __name__ == "__main__":
    setup_model()