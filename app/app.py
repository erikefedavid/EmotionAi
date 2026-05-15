import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify
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

def gen_frames():
    cap = cv2.VideoCapture(0)
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame_count += 1
            # Convert to gray for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Only predict every 5 frames to keep the video smooth and fast
            if frame_count % 5 == 0:
                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                    if model:
                        roi = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
                        roi = roi.astype('float') / 255.0
                        roi = img_to_array(roi)
                        roi = np.expand_dims(roi, axis=0)

                        prediction = model.predict(roi, verbose=0)[0]
                        label = EMOTIONS[prediction.argmax()]
                        confidence = float(np.max(prediction))
                        
                        current_state["emotion"] = label
                        current_state["confidence"] = confidence * 100
                
                if len(faces) == 0:
                    current_state["emotion"] = "Scanning..."
                    current_state["confidence"] = 0

            # Draw on EVERY frame for smooth visual feedback
            for (x, y, w, h) in faces:
                label = current_state["emotion"]
                conf = current_state["confidence"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (99, 102, 241), 2)
                if label != "Scanning...":
                    cv2.rectangle(frame, (x, y-40), (x+w, y), (99, 102, 241), -1)
                    cv2.putText(frame, f"{label} {conf:.0f}%", (x+5, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
