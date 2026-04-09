from pathlib import Path
import io

from flask import Flask, request, jsonify, send_from_directory
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

ROOF_FILTER_MODEL_PATH = BASE_DIR / "models" / "efficientnet_b2_roof_filter_best.pth"
RUST_MODEL_PATH = BASE_DIR / "models" / "efficientnet_b2_roofrust_best_b2_mixup.pth"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}

ROOF_THRESHOLD = 70.0
LOW_CONFIDENCE_THRESHOLD = 70.0
REJECT_THRESHOLD = 40.0

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

if not ROOF_FILTER_MODEL_PATH.exists():
    raise FileNotFoundError(f"Roof filter checkpoint not found: {ROOF_FILTER_MODEL_PATH}")

if not RUST_MODEL_PATH.exists():
    raise FileNotFoundError(f"Rust classifier checkpoint not found: {RUST_MODEL_PATH}")

if not (STATIC_DIR / "index.html").exists():
    raise FileNotFoundError(f"index.html not found in: {STATIC_DIR}")

DISPLAY_ORDER = [
    "No Rust",
    "Slightly Visible Rust",
    "Visible Rust",
    "Heavy Visible Rust"
]

CLASS_DETAILS = {
    "No Rust": {
        "iso_basis": "Closest to Grade A, where the steel surface is largely covered with adhering mill scale and has little or no rust.",
        "recommendation": [
            "Your roof looks in good condition.",
            "No immediate repair is needed.",
            "Keep the roof clean and dry.",
            "Check it regularly, especially after strong rain or storms.",
            "Apply protective coating during scheduled maintenance to help prevent future rust."
        ]
    },
    "Slightly Visible Rust": {
        "iso_basis": "Closest to Grade B, where rusting has begun and mill scale has started to flake.",
        "recommendation": [
            "Early signs of rust are starting to appear.",
            "Clean the affected area as soon as possible.",
            "Remove dirt, loose rust, and debris.",
            "Apply primer or protective paint to stop the rust from spreading.",
            "Monitor the area regularly to make sure the damage does not get worse."
        ]
    },
    "Visible Rust": {
        "iso_basis": "Closest to Grade C, where mill scale has rusted away or can be scraped off and slight pitting is visible.",
        "recommendation": [
            "Rust is already noticeable and should not be ignored.",
            "Schedule maintenance soon to avoid bigger damage.",
            "Clean and prepare the surface properly before repainting or recoating.",
            "Check nearby screws, joints, and fasteners because rust often spreads around these areas.",
            "If the rust covers a larger area, ask a roofing professional to inspect it."
        ]
    },
    "Heavy Visible Rust": {
        "iso_basis": "Closest to Grade D, where mill scale has rusted away and general pitting is visible.",
        "recommendation": [
            "The roof shows serious rust and may already be weakening.",
            "Urgent repair is recommended.",
            "Have the roof inspected for deep corrosion, holes, or possible structural damage.",
            "Heavy rust should be removed thoroughly before any recoating is done.",
            "If the damage is severe, some roof parts may need professional restoration or replacement."
        ]
    }
}

preprocess = EfficientNet_B2_Weights.DEFAULT.transforms()

# =========================
# LOAD ROOF FILTER MODEL
# =========================
roof_checkpoint = torch.load(ROOF_FILTER_MODEL_PATH, map_location=device, weights_only=False)
roof_class_names = roof_checkpoint["class_names"]
roof_num_classes = len(roof_class_names)

roof_model = efficientnet_b2(weights=None)
roof_in_features = roof_model.classifier[1].in_features
roof_model.classifier[1] = nn.Linear(roof_in_features, roof_num_classes)
roof_model.load_state_dict(roof_checkpoint["model_state_dict"])
roof_model = roof_model.to(device)
roof_model.eval()

# =========================
# LOAD RUST MODEL
# =========================
rust_checkpoint = torch.load(RUST_MODEL_PATH, map_location=device, weights_only=False)
rust_class_names = rust_checkpoint["class_names"]
rust_num_classes = len(rust_class_names)

rust_model = efficientnet_b2(weights=None)
rust_in_features = rust_model.classifier[1].in_features
rust_model.classifier[1] = nn.Linear(rust_in_features, rust_num_classes)
rust_model.load_state_dict(rust_checkpoint["model_state_dict"])
rust_model = rust_model.to(device)
rust_model.eval()

def run_roof_filter(image: Image.Image):
    image = image.convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = roof_model(input_tensor)
        probs = torch.softmax(outputs[0], dim=0)

    result = {roof_class_names[i]: float(probs[i]) for i in range(roof_num_classes)}
    predicted_class = max(result, key=result.get)
    confidence = result[predicted_class] * 100

    return predicted_class, confidence, result

def run_rust_classifier(image: Image.Image):
    image = image.convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = rust_model(input_tensor)
        probs = torch.softmax(outputs[0], dim=0)

    raw_result = {rust_class_names[i]: float(probs[i]) for i in range(rust_num_classes)}
    final_class = max(raw_result, key=raw_result.get)
    confidence = raw_result[final_class] * 100

    ordered_result = {}
    for cls in DISPLAY_ORDER:
        ordered_result[cls] = raw_result.get(cls, 0.0)

    if confidence < REJECT_THRESHOLD:
        return {
            "accepted": False,
            "low_confidence": False,
            "error": "The image is a roof, but the rust severity result is too uncertain. Please upload a clearer roof image."
        }

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return {
            "accepted": True,
            "low_confidence": True,
            "warning": "The image was detected as a roof, but the rust severity prediction is not highly confident. Please review the result carefully or try another clearer image.",
            "predictions": ordered_result,
            "final_classification": final_class,
            "confidence": round(confidence, 2),
            "iso_basis": CLASS_DETAILS[final_class]["iso_basis"],
            "recommendation": CLASS_DETAILS[final_class]["recommendation"]
        }

    return {
        "accepted": True,
        "low_confidence": False,
        "warning": "",
        "predictions": ordered_result,
        "final_classification": final_class,
        "confidence": round(confidence, 2),
        "iso_basis": CLASS_DETAILS[final_class]["iso_basis"],
        "recommendation": CLASS_DETAILS[final_class]["recommendation"]
    }

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": "Unsupported file type. Please upload JPG, JPEG, PNG, BMP, or WEBP."
        }), 400

    try:
        image_bytes = file.read()

        if not image_bytes:
            return jsonify({"error": "Uploaded file is empty"}), 400

        image_stream = io.BytesIO(image_bytes)

        test_img = Image.open(image_stream)
        test_img.verify()

        image_stream.seek(0)
        image = Image.open(image_stream).convert("RGB")

        roof_class, roof_confidence, roof_probs = run_roof_filter(image)

        roof_label_map = {name.lower(): name for name in roof_class_names}
        predicted_roof_label = roof_label_map.get("roof", "roof")

        if roof_class != predicted_roof_label or roof_confidence < ROOF_THRESHOLD:
            return jsonify({
                "error": "Please upload a roof image only. The uploaded image was detected as non-roof or too uncertain.",
                "roof_filter_prediction": roof_class,
                "roof_filter_confidence": round(roof_confidence, 2)
            }), 400

        rust_result = run_rust_classifier(image)

        if not rust_result["accepted"]:
            return jsonify({
                "error": rust_result["error"],
                "roof_filter_prediction": roof_class,
                "roof_filter_confidence": round(roof_confidence, 2)
            }), 400

        rust_result["roof_filter_prediction"] = roof_class
        rust_result["roof_filter_confidence"] = round(roof_confidence, 2)

        return jsonify(rust_result)

    except UnidentifiedImageError:
        return jsonify({
            "error": "Invalid or corrupted image file. Please upload a valid image."
        }), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)