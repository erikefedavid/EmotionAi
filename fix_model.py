# fix_model.py
import os
import h5py
import json
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
print("TF version:", tf.__version__)

def fix(obj):
    if isinstance(obj, dict):
        # THE CORE FIX: batch_shape -> shape
        if 'batch_shape' in obj:
            shape = obj.pop('batch_shape')
            obj['shape'] = shape[1:]  # strip batch dim
        obj.pop('optional', None)
        obj.pop('registered_name', None)
        obj.pop('quantization_config', None)
        obj.pop('build_input_shape', None)
        if isinstance(obj.get('dtype'), dict):
            obj['dtype'] = obj['dtype'].get('config', {}).get('name', 'float32')
        for v in list(obj.values()):
            fix(v)
    elif isinstance(obj, list):
        for item in obj:
            fix(item)

# Read, patch, write config back into the h5 file
with h5py.File('models/simple_cnn.h5', 'r+') as f:
    raw = f.attrs.get('model_config')
    if hasattr(raw, 'decode'):
        raw = raw.decode('utf-8')
    config = json.loads(raw)
    fix(config)
    f.attrs['model_config'] = json.dumps(config).encode('utf-8')
    print("Config patched successfully")

# Now load it
model = tf.keras.models.load_model('models/simple_cnn.h5', compile=False)
print("SUCCESS! Model loaded")
model.save('models/simple_cnn_fixed.keras')
print("Saved: models/simple_cnn_fixed.keras")