import os
import sys
from pathlib import Path

# Memory optimization for Render
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Robust pathing for Render
BASE_DIR = Path(__file__).parent
app = Flask(__name__, template_folder=str(BASE_DIR / 'templates'))

# Global variables to track loading status
model_loading_error = "Not attempted"
face_loading_error = "Not attempted"

# Load Face Classifier (Haar Cascade)
try:
    CASCADE_PATH = BASE_DIR / 'haarcascade_frontalface_default.xml'
    if not CASCADE_PATH.exists():
        CASCADE_PATH = BASE_DIR.parent / 'app' / 'haarcascade_frontalface_default.xml'

    face_classifier = cv2.CascadeClassifier(str(CASCADE_PATH))
    if face_classifier.empty():
        raise Exception(f"Classifier is empty at {CASCADE_PATH}")
    print(f"SUCCESS: Face Classifier Loaded from: {CASCADE_PATH}")
    face_loading_error = "None"
except Exception as e:
    face_loading_error = str(e)
    print(f"ERROR loading face classifier: {face_loading_error}")
    face_classifier = None

# Load Emotion Model
try:
    model_found = False
    # Check all files in BASE_DIR for any .h5 file
    for file in os.listdir(BASE_DIR):
        if file.lower() == 'simple_cnn.h5':
            MODEL_PATH = BASE_DIR / file
            # Bypassing version mismatch errors with compile=False
            model = load_model(str(MODEL_PATH), compile=False)
            print(f"SUCCESS: Emotion Model Loaded from: {MODEL_PATH}")
            model_loading_error = "None"
            model_found = True
            break
    
    if not model_found:
        # Check root models folder too
        ROOT_MODELS = BASE_DIR.parent / 'models'
        if ROOT_MODELS.exists():
            for file in os.listdir(ROOT_MODELS):
                if file.lower() == 'simple_cnn.h5':
                    MODEL_PATH = ROOT_MODELS / file
                    # Bypassing version mismatch errors with compile=False
                    model = load_model(str(MODEL_PATH), compile=False)
                    print(f"SUCCESS: Emotion Model Loaded from: {MODEL_PATH}")
                    model_loading_error = "None"
                    model_found = True
                    break
                    
    if not model_found:
        model_loading_error = f"File simple_cnn.h5 not found in {BASE_DIR} or {BASE_DIR.parent / 'models'}. Files present in {BASE_DIR}: {os.listdir(BASE_DIR)}"
        model = None
except Exception as e:
    model_loading_error = f"Crash during load: {str(e)}"
    print(f"ERROR loading model: {model_loading_error}")
    model = None

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"]

@app.route('/debug')
def debug_files():
    import tensorflow as tf
    import keras
    files_info = []
    for root, dirs, files in os.walk('.'):
        for name in files:
            p = os.path.join(root, name)
            files_info.append(f"{p} ({os.path.getsize(p)} bytes)")
    return jsonify({
        "tf_version": tf.__version__,
        "keras_version": keras.__version__,
        "cwd": os.getcwd(),
        "face_error": face_loading_error,
        "model_error": model_loading_error,
        "files_found": files_info
    })

def get_recommendation(emotion):
    recommendations = {
        "Happy": "You're glowing! Perfect time to tackle your hardest task or share some joy. 🌟",
        "Sad": "It's okay to feel down. How about a cup of tea and some relaxing Lo-Fi? ☕",
        "Angry": "Deep breaths... counting to 10 often helps. Maybe some calm jazz? 🎷",
        "Fear": "You're safe. Try some grounding exercises or talk to a friend. 🛡️",
        "Surprise": "Wow! What a moment! Keep that curiosity alive. 🔍",
        "Disgust": "Something's not right? Take a break and refresh your environment. 🍃",
        "Scanning...": "Position your face in the frame to begin analysis. 🔍"
    }
    return recommendations.get(emotion, "Analyzing your mood...")

@app.route('/status')
def status():
    return jsonify({
        "emotion": current_state["emotion"],
        "confidence": f"{current_state['confidence']:.1f}%",
        "suggestion": get_recommendation(current_state["emotion"])
    })

import base64
import io
from PIL import Image

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # CHECK MODELS
        if face_classifier is None:
            return jsonify({"emotion": "Detector missing", "confidence": 0, "suggestion": f"Error: {face_loading_error}"})
        if model is None:
            return jsonify({"emotion": "Model missing", "confidence": 0, "suggestion": f"Error: {model_loading_error}"})

        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Convert to OpenCV format
        frame = np.array(img)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # HIGH SENSITIVITY DETECTION
        faces = face_classifier.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=3, 
            minSize=(20, 20)
        )
        
        print(f"DEBUG: Found {len(faces)} faces in frame")
        
        if len(faces) == 0:
            return jsonify({"emotion": "Searching...", "confidence": 0, "suggestion": "No face detected. Get closer and look directly at the camera."})

        # Process the first face
        (x, y, w, h) = faces[0]
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        
        if model:
            roi = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
            roi = roi.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = model.predict(roi, verbose=0)[0]
            label = EMOTIONS[prediction.argmax()]
            confidence = float(np.max(prediction)) * 100
            
            return jsonify({
                "emotion": label,
                "confidence": f"{confidence:.1f}%",
                "suggestion": get_recommendation(label)
            })
        
        return jsonify({"error": "Model not loaded"}), 500
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
