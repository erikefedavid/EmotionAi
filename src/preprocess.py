import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import urllib.request

# Emotion mappings
EMOTIONS = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral' # Note: PRD mentions 6 emotion categories but FER2013 has 7. We will use all 7 or remap. The PRD says "classify into one of six emotion categories: Anger, Disgust, Fear, Happiness, Sadness, and Surprise." So we need to ignore neutral or map it. Wait, PRD: "Anger, Disgust, Fear, Happiness, Sadness, and Surprise". We will skip class 6 (neutral) if it's strictly 6.
}

def ensure_haar_cascade():
    # Download Haar Cascade if it doesn't exist
    haar_path = Path(__file__).parent.parent / 'app' / 'haar' / 'haarcascade_frontalface_default.xml'
    haar_path.parent.mkdir(parents=True, exist_ok=True)
    if not haar_path.exists():
        print("Downloading Haar Cascade...")
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        urllib.request.urlretrieve(url, haar_path)
    return str(haar_path)

def decode_pixels(pixel_string):
    pixels = np.array(pixel_string.split(' '), dtype='uint8')
    return pixels.reshape((48, 48, 1))

def apply_clahe(img_gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img_gray)

def augment_image(img):
    # Random horizontal flip
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)
        
    # Random rotation
    angle = np.random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((24, 24), angle, 1.0)
    img = cv2.warpAffine(img, M, (48, 48))
    
    # Brightness/Contrast
    alpha = np.random.uniform(0.8, 1.2) # contrast
    beta = np.random.uniform(-20, 20)   # brightness
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    return img

def preprocess_and_save():
    root_dir = Path(__file__).parent.parent
    csv_path = root_dir / 'data' / 'raw' / 'fer2013.csv'
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Please run download_data.py first.")
        return
        
    haar_cascade_path = ensure_haar_cascade()
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)
    
    print("Reading CSV...")
    df = pd.read_csv(csv_path)
    
    # ImageNet stats for normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    skip_count = 0
    
    # Setup directories
    for usage in ['train', 'val', 'test']:
        for emotion_name in EMOTIONS.values():
            if emotion_name == 'neutral':
                continue # PRD says 6 categories, skipping neutral
            (root_dir / 'data' / usage / emotion_name).mkdir(parents=True, exist_ok=True)
            
    print("Processing images...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        emotion_idx = row['emotion']
        if emotion_idx not in EMOTIONS or EMOTIONS[emotion_idx] == 'neutral':
            continue # Skip neutral
            
        emotion_name = EMOTIONS[emotion_idx]
        usage = row['Usage']
        
        # Map usage to folder names
        if usage == 'Training':
            split = 'train'
        elif usage == 'PublicTest':
            split = 'val'
        elif usage == 'PrivateTest':
            split = 'test'
        else:
            continue
            
        img_gray = decode_pixels(row['pixels'])
        
        # Face detection
        faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
        
        if len(faces) == 0:
            skip_count += 1
            # If no face detected, we will just use the whole image as per fallback
            face_img = img_gray
        else:
            # Crop largest bounding box
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            x, y, w, h = faces[0]
            face_img = img_gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (48, 48))
            face_img = np.expand_dims(face_img, axis=-1)
            
        # Reshape to (48, 48) for processing
        face_img_2d = face_img.reshape((48, 48))
        
        # CLAHE
        face_img_clahe = apply_clahe(face_img_2d)
        
        # Convert to RGB by stacking
        img_rgb = cv2.cvtColor(face_img_clahe, cv2.COLOR_GRAY2RGB)
        
        if split == 'train':
            # Augment
            img_rgb = augment_image(img_rgb)
            
        # Normalize (this is typically done in the tf.data pipeline or keras generator, 
        # but PRD Step 5 says to do it here. However, since we must save as .jpg for ImageDataGenerator,
        # saving normalized float values [0,1] to JPG will lose precision or clip. 
        # We will save the visual RGB images and apply normalization during dataset loading in train.py).
        # We will just save the 8-bit RGB image here.
        
        save_path = root_dir / 'data' / split / emotion_name / f"{idx}.jpg"
        cv2.imwrite(str(save_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    print(f"Preprocessing complete. Skipped/Fallback face detection for {skip_count} images.")

if __name__ == '__main__':
    preprocess_and_save()
