import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Path to your current model
model_path = 'app/simple_cnn.h5'
new_model_path = 'app/simple_cnn_legacy.h5'

print(f"Loading model from {model_path}...")
# Load the model (using compile=False to avoid issues)
model = load_model(model_path, compile=False)

print("Converting to Legacy Format...")
# Save in the older H5 format explicitly
model.save(new_model_path, save_format='h5')

print(f"DONE! Created: {new_model_path}")
print("Now, I will replace the old model with this one.")

# Swap them
os.replace(new_model_path, model_path)
print("SUCCESS: Your model is now Cloud-Ready!")
