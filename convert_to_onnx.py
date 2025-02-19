import torch
from transformers import ViTImageProcessor, ViTForImageClassification
import onnx


# Load model and image processor
model_name = "nateraw/vit-age-classifier"
model = ViTForImageClassification.from_pretrained(model_name)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)

# Set model to evaluation mode
model.eval()

# Get correct image size
image_size = feature_extractor.size["height"]  # Default: 224
dummy_image = torch.randn(1, 3, image_size, image_size)

# Convert to ONNX with updated opset version (14+)
onnx_path = "vit_age_classifier.onnx"
torch.onnx.export(
    model,
    (dummy_image,),
    onnx_path,
    input_names=["pixel_values"],
    output_names=["logits"],
    dynamic_axes={"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}},
    opset_version=14,
)

print(f"Model successfully converted to ONNX and saved at: {onnx_path}")
