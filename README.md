# Facial Emotion Recognition (FER) System

This repository contains the full pipeline for training and deploying a Facial Emotion Recognition system based on ResNet-50 and FER2013 dataset. It was developed as a final year BSc project.

## Requirements
- Python 3.10+ (Tested on Python 3.12)
- Virtual Environment

## Setup Instructions
1. Create and activate the virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\Activate.ps1
   # On Mac/Linux:
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up Kaggle API:
   Ensure your `kaggle.json` credentials are placed in `~/.kaggle/` (or `C:\Users\<user>\.kaggle\`).

4. Download the dataset:
   ```bash
   python src/download_data.py
   ```
   This will download and extract `fer2013.csv` into `data/raw/`.

## Running the Pipeline
1. **Preprocessing**:
   ```bash
   python src/preprocess.py
   ```
   Converts the CSV to organized folders under `data/processed/` and `data/train/`, `val/`, `test/`.

2. **Training**:
   *(Note: Training ResNet-50 requires significant computation. If you do not have a dedicated GPU, it is recommended to run `src/train.py` on Google Colab or Kaggle Kernels.)*
   ```bash
   python src/train.py
   python src/train_baselines.py
   ```

3. **Evaluation**:
   ```bash
   python src/evaluate.py
   ```

4. **Web App Demo**:
   ```bash
   python app/app.py
   ```
   Visit `http://localhost:5000` to interact with the model via image upload or webcam feed.
