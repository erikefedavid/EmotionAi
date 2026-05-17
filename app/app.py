import os
import sys
import base64
import io
from pathlib import Path

# Must be set BEFORE any TF/Keras import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from PIL import Image

# Use tf_keras explicitly — avoids Keras 3 batch_shape incompatibility
import tf_keras
from tf_keras.models import load_model
from tf_keras.preprocessing.image import img_to_array

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
ROOT_DIR = BASE_DIR.parent

app = Flask(__name__, template_folder=str(BASE_DIR / 'templates'))

# ── Constants ────────────────────────────────────────────────────────────────
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"]

RECOMMENDATIONS = {
    "Happy":     "You're glowing! Perfect time to tackle your hardest task. 🌟",
    "Sad":       "It's okay to feel down. How about some relaxing Lo-Fi? ☕",
    "Angry":     "Deep breaths... counting to 10 often helps. 🎷",
    "Fear":      "You're safe. Try some grounding exercises. 🛡️",
    "Surprise":  "Wow! Keep that curiosity alive. 🔍",
    "Disgust":   "Take a break and refresh your environment. 🍃",
    "Scanning...": "Position your face in the frame to begin. 🔍"
}

# ── Load Face Classifier ─────────────────────────────────────────────────────
face_classifier = None
face_error = "Not loaded"

CASCADE_PATHS = [
    BASE_DIR / 'haarcascade_frontalface_default.xml',
    BASE_DIR.parent / 'app' / 'haarcascade_frontalface_default.xml',
]

for path in CASCADE_PATHS:
    if path.exists():
        clf = cv2.CascadeClassifier(str(path))
        if not clf.empty():
            face_classifier = clf
            face_error = None
            print(f"✓ Face classifier loaded: {path}")
            break

if face_classifier is None:
    face_error = f"Haar cascade not found. Searched: {[str(p) for p in CASCADE_PATHS]}"
    print(f"✗ {face_error}")

# ── Load Emotion Model ───────────────────────────────────────────────────────
model = None
model_error = "Not loaded"

MODEL_SEARCH_DIRS = [BASE_DIR, ROOT_DIR / 'models']

for search_dir in MODEL_SEARCH_DIRS:
    if not search_dir.exists():
        continue
    for fname in os.listdir(search_dir):
        if fname.lower() == 'simple_cnn_fixed.keras':
            model_path = search_dir / fname
            try:
                model = load_model(str(model_path), compile=False)
                model_error = None
                print(f"✓ Model loaded: {model_path}")
            except Exception as e:
                model_error = f"Load failed: {e}"
                print(f"✗ {model_error}")
            break
    if model is not None:
        break

if model is None and model_error == "Not loaded":
    model_error = f"simple_cnn.h5 not found in: {[str(d) for d in MODEL_SEARCH_DIRS]}"
    print(f"✗ {model_error}")

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/debug')
def debug():
    import tensorflow as tf
    files = []
    for root, _, fnames in os.walk('.'):
        for name in fnames:
            p = os.path.join(root, name)
            files.append(f"{p} ({os.path.getsize(p):,} bytes)")
    return jsonify({
        "tf_version":    tf.__version__,
        "tf_keras_version": tf_keras.__version__,
        "cwd":           os.getcwd(),
        "model_error":   model_error,
        "face_error":    face_error,
        "files":         files
    })


@app.route('/predict', methods=['POST'])
def predict():
    # Guard checks
    if face_classifier is None:
        return jsonify({"emotion": "Detector missing", "confidence": 0,
                        "suggestion": f"Error: {face_error}"}), 503
    if model is None:
        return jsonify({"emotion": "Model missing", "confidence": 0,
                        "suggestion": f"Error: {model_error}"}), 503

    try:
        data = request.get_json(force=True)
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)

        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        frame = np.array(img)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        faces = face_classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20)
        )

        if len(faces) == 0:
            return jsonify({
                "emotion": "Searching...",
                "confidence": 0,
                "suggestion": "No face detected. Move closer and face the camera."
            })

        x, y, w, h = faces[0]
        roi = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, (48, 48))
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        roi = roi.astype('float32') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        prediction = model.predict(roi, verbose=0)[0]
        emotion = EMOTIONS[prediction.argmax()]
        confidence = float(np.max(prediction)) * 100

        return jsonify({
            "emotion":    emotion,
            "confidence": f"{confidence:.1f}%",
            "suggestion": RECOMMENDATIONS.get(emotion, "Analyzing...")
        })

    except Exception as e:
        print(f"[predict error] {e}")
        return jsonify({"error": str(e)}), 400


# ── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port, debug=False)