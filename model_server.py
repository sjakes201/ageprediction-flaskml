import io
import numpy as np
from PIL import Image
import onnxruntime
from flask_ml.flask_ml_server import MLServer, load_file_as_string
from flask_ml.flask_ml_server.models import ResponseBody
from flask import request, jsonify

# Preprocessing
def preprocess_image(image_bytes, target_size=224):
    """
    Preprocess the input image:
      - Load and convert to RGB
      - Resize to target dimensions (default 224x224)
      - Convert to a numpy array and scale pixel values to [0, 1]
      - Normalize using mean and std of [0.5, 0.5, 0.5] (as float32)
      - Rearrange the array to channel-first (C, H, W) and add a batch dimension.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((target_size, target_size))
    image_np = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    image_np = (image_np - mean) / std
    image_np = np.transpose(image_np, (2, 0, 1))
    image_np = np.expand_dims(image_np, axis=0)
    return image_np

# Postprocessing
def postprocess_output(logits):
    """
    Apply softmax to the raw logits and return predicted classes for the batch.
    """
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    predicted_classes = np.argmax(probabilities, axis=1)
    return predicted_classes, probabilities

# Mapping from classification ids to age ranges
id2label = {
    0: "0-2",
    1: "3-9",
    2: "10-19",
    3: "20-29",
    4: "30-39",
    5: "40-49",
    6: "50-59",
    7: "60-69",
    8: "more than 70"
}

# Load ONNX model
onnx_model_path = "vit_age_classifier.onnx"
onnx_session = onnxruntime.InferenceSession(onnx_model_path)

# ML server setup
server = MLServer(__name__)
server.add_app_metadata(
    name="Vit Age Classifier",
    author="",
    version="1.0.0",
    info="Age classifier using ViT exported to ONNX."
)

# Custom Endpoint for Batch Image Upload 
@server.app.route("/predict_age", methods=["POST"])
def predict_age_plain():
    # Get a list of files uploaded with key "image"
    files = request.files.getlist("image")
    if not files or len(files) == 0:
        return jsonify({"error": "No image files provided."}), 400

    batch_images = []
    for file in files:
        if file.filename == "":
            continue
        # Read file bytes and preprocess each image.
        image_bytes = file.read()
        processed = preprocess_image(image_bytes)  # shape: (1, C, H, W)
        batch_images.append(processed)

    if len(batch_images) == 0:
        return jsonify({"error": "No valid images provided."}), 400

    # Concatenate all preprocessed images into a batch (shape: (batch, C, H, W))
    batch = np.concatenate(batch_images, axis=0)

    try:
        outputs = onnx_session.run(None, {"pixel_values": batch})
        logits = outputs[0]  # shape: (batch, num_classes)
        predicted_classes, _ = postprocess_output(logits)
        # Map each predicted class to its label
        predicted_labels = [id2label.get(cls, "Unknown") for cls in predicted_classes]
        return jsonify({
            "predicted_labels": predicted_labels
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Vit Age Classifier Server")
    parser.add_argument("--port", type=int, default=5000, help="Port number to run the server")
    args = parser.parse_args()
    server.run(port=args.port)

