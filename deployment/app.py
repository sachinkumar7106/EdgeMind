
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
MODEL_PATH = BASE_DIR / "models" / "student" / "best_distilled_student.pth"

sys.path.insert(0, str(SRC_DIR))

from student import CNNStudent

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]}})

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
])


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    model = CNNStudent(num_classes=len(CLASSES)).to(DEVICE)

    try:
        checkpoint = torch.load(
            MODEL_PATH, map_location=DEVICE, weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


model = load_model()
print(f"EdgeMind model loaded on {DEVICE}")


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model": "distilled CNN student",
        "device": str(DEVICE),
    })


@app.post("/api/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Please upload an image."}), 400

    file = request.files["image"]

    if not file.filename:
        return jsonify({"error": "No image selected."}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify({
            "error": "The uploaded file is not a valid image."
        }), 400

    try:
        tensor = preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            logits = model(tensor)
            probabilities = F.softmax(logits, dim=1)
            confidence, predicted_index = probabilities.max(dim=1)
  
        index = int(predicted_index.item())

        # Get the probability for each of the 10 classes
        all_probabilities = {
            CLASSES[i]: round(float(probabilities[0, i].item()) * 100, 2)
            for i in range(len(CLASSES))
        }

        return jsonify({
            "prediction": CLASSES[index],
            "confidence": round(float(confidence.item()) * 100, 2),
            "probabilities": all_probabilities,
        })

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed. Check the model and server logs."
        }), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
