import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from pathlib import Path

app = Flask(__name__)

# Load Model
MODEL_PATH = Path(__file__).parent.parent / 'models' / 'simple_cnn.h5'
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Check if model exists
if not MODEL_PATH.exists():
    print(f"WARNING: Model not found at {MODEL_PATH}. App will start but prediction will fail.")
    model = None
else:
    model = load_model(str(MODEL_PATH))

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"]
current_state = {"emotion": "Scanning...", "confidence": 0}

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
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Convert to OpenCV format
        frame = np.array(img)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = face_classifier.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) == 0:
            return jsonify({"emotion": "Scanning...", "confidence": 0, "suggestion": get_recommendation("Scanning...")})

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
