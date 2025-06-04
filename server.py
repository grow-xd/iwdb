import warnings
from flask import Flask, request, jsonify
import pickle
import re
import numpy as np
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from fastai.vision.all import *
from werkzeug.utils import secure_filename
import requests
from PIL import Image
from io import BytesIO
from flask_cors import CORS

# === Config ===
TEXT_MODEL_PATH = "text-ml/random_forest.pkl"
VECTORIZER_PATH = "text-ml/tfidf_vectorizer.pkl"  
IMAGE_MODEL_PATH = "image-cnn/resnet34_nsfw.pkl"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# === App Setup ===
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Suppress FastAI warnings ===
warnings.filterwarnings("ignore", category=UserWarning, module="fastai.learner")

# === Load Text Model and Vectorizer ===
with open(TEXT_MODEL_PATH, "rb") as f:
    text_model = pickle.load(f)
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# === Load FastAI Image Model ===
learn = load_learner(IMAGE_MODEL_PATH)

# === Label Map for Text ===
LABEL_MAP = {0: "Hate Speech", 1: "Offensive Language", 2: "Neither"}

# === Helpers ===
def preprocess_text(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    stop_words.update(["#ff", "ff", "rt"])

    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()

    tokens = text.split()
    tokens = [stemmer.stem(tok) for tok in tokens if tok not in stop_words]
    return " ".join(tokens)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# === Routes ===

@app.route('/predict-text', methods=['POST'])
def predict_text():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    domain = data["domain"]
    raw_text = data["text"]
    processed = preprocess_text(raw_text)
    vector = vectorizer.transform([processed])

    if hasattr(text_model, "predict_proba"):
        probs = text_model.predict_proba(vector)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
    elif hasattr(text_model, "decision_function"):
        scores = text_model.decision_function(vector)[0]
        idx = int(np.argmax(scores))
        confidence = float(scores[idx])
    else:
        idx = int(text_model.predict(vector)[0])
        confidence = None

    # Call an external API to localhost:8000/api/items
    try:
        response = requests.post("http://localhost:8000/api/items/", json={
            "url": domain,
            "score": confidence,
            "class": "good" if idx == 2 else "bad"
        })
        if response.status_code == 200:
            api_result = response.json()
        else:
            api_result = {"error": f"API call failed with status code {response.status_code}"}
    except Exception as e:
        api_result = {"error": f"API call failed: {str(e)}"}

    return jsonify({
        "input": raw_text,
        "predicted_label": LABEL_MAP.get(idx, str(idx)),
        "class_index": idx,
        "confidence": confidence
    })


@app.route('/predict-image', methods=['POST'])
def predict_image():
    try:
        # Case 1: Direct image URL
        if 'image_url' in request.form:
            image_url = request.form['image_url']
            domain = request.form.get("domain", "")
            response = requests.get(image_url)
            image = PILImage.create(BytesIO(response.content))

        # Case 2: Uploaded image blob (from <video>)
        elif 'image' in request.files:
            domain = request.form.get("domain", "")
            image_file = request.files['image']
            image = PILImage.create(image_file)

        else:
            return jsonify({"error": "Missing 'image_url' or 'image' in form-data"}), 400

        pred_class, pred_idx, probs = learn.predict(image)
        confidence = float(probs[pred_idx])

        # Call an external API to localhost:8000/api/items
        try:
            response = requests.post("http://localhost:8000/api/items/", json={
                "url": domain,
                "score": confidence,
                "class": "good" if (pred_class=="drawing" or pred_class=="neutral") else "bad"
            })
            if response.status_code == 200:
                api_result = response.json()
            else:
                api_result = {"error": f"API call failed with status code {response.status_code}"}
        except Exception as e:
            api_result = {"error": f"API call failed: {str(e)}"}

        return jsonify({
            "predicted_class": str(pred_class),
            "class_index": int(pred_idx),
            "confidence": confidence
        })
        

    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500


# === Run App ===
if __name__ == "__main__":
    app.run(debug=True, port=5000)
