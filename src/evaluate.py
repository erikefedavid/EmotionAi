import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.utils import get_root_dir

def evaluate_model(model_path, test_gen, classes):
    print(f"Evaluating {model_path.name}...")
    model = load_model(str(model_path))
    
    # Predict
    preds = model.predict(test_gen)
    pred_classes = np.argmax(preds, axis=1)
    true_classes = test_gen.classes
    
    # Metrics
    report = classification_report(true_classes, pred_classes, target_names=classes, output_dict=True)
    cm = confusion_matrix(true_classes, pred_classes)
    weighted_f1 = f1_score(true_classes, pred_classes, average='weighted')
    
    return report, cm, weighted_f1

def main():
    root_dir = get_root_dir()
    test_dir = root_dir / 'data' / 'test'
    
    # PRD requires 48x48x3 images
    # Using ResNet50 preprocess_input for the main model, and 1/255 for baselines? 
    # For simplicity in evaluation script, we will use a generic generator and handle preprocess per model if needed.
    # However, since we are loading .h5, we should be consistent with training.
    
    datagen = ImageDataGenerator(rescale=1./255) # Default for evaluation comparison
    
    test_gen = datagen.flow_from_directory(
        test_dir,
        target_size=(48, 48),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    classes = list(test_gen.class_indices.keys())
    models_dir = root_dir / 'models'
    results_dir = root_dir / 'results'
    
    summary = {}
    
    for model_file in models_dir.glob('*.h5'):
        model_name = model_file.stem
        report, cm, f1 = evaluate_model(model_file, test_gen, classes)
        
        summary[model_name] = {
            'weighted_f1': f1,
            'accuracy': report['accuracy'],
            'report': report
        }
        
        # Plot Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(results_dir / 'figures' / f'confusion_matrix_{model_name}.png', dpi=150)
        plt.close()
        
    # Save JSON summary
    with open(results_dir / 'metrics_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
        
    print("Evaluation complete. Summary saved to results/metrics_summary.json")

if __name__ == '__main__':
    main()
