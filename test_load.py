import os
import sys
import h5py
import json
import traceback

os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_PATH = 'app/simple_cnn.h5'

print("TF Version:", tf.__version__)
try:
    import keras
    print("Keras Version:", keras.__version__)
except Exception:
    pass

try:
    print("Attempting to load model normally...")
    model = load_model(MODEL_PATH, compile=False)
    print("Success loading normally!")
except Exception as e:
    print("Normal load failed with:")
    traceback.print_exc()
    
    print("\nApplying custom fix...")
    try:
        with h5py.File(MODEL_PATH, 'r+') as f:
            model_config = f.attrs.get('model_config')
            if model_config:
                if hasattr(model_config, 'decode'):
                    model_config = model_config.decode('utf-8')
                
                config = json.loads(model_config)
                
                # Check sequential config
                if isinstance(config, dict) and 'config' in config:
                    # Strip Sequential's build_input_shape or registered_name
                    if isinstance(config['config'], dict):
                        config['config'].pop('registered_name', None)
                        config['config'].pop('build_input_shape', None)
                
                def fix_layer(layer):
                    if not isinstance(layer, dict):
                        return
                    if 'config' in layer:
                        # Strip K3 specific keys
                        layer['config'].pop('batch_shape', None)
                        layer['config'].pop('registered_name', None)
                        layer['config'].pop('optional', None)
                        layer['config'].pop('quantization_config', None)
                        
                        # Handle DTypePolicy dicts
                        dtype = layer['config'].get('dtype')
                        if isinstance(dtype, dict) and 'config' in dtype:
                            layer['config']['dtype'] = dtype['config'].get('name', 'float32')
                        
                        # Handle initializers/regularizers that are dicts with 'module'
                        for key in ['kernel_initializer', 'bias_initializer', 'kernel_regularizer', 'bias_regularizer', 'activity_regularizer']:
                            val = layer['config'].get(key)
                            if isinstance(val, dict) and 'class_name' in val:
                                # Simplify to Keras 2 style dict
                                layer['config'][key] = {
                                    'class_name': val['class_name'],
                                    'config': val.get('config', {})
                                }
                                
                    if 'layers' in layer and isinstance(layer['layers'], list):
                        for sub in layer['layers']:
                            fix_layer(sub)
                
                fix_layer(config)
                f.attrs['model_config'] = json.dumps(config).encode('utf-8')
                print("Fix written to file.")
    except Exception as fe:
        print("Failed to apply fix:")
        traceback.print_exc()

    print("\nAttempting to load model again after fix...")
    try:
        model = load_model(MODEL_PATH, compile=False)
        print("Success loading after fix!")
    except Exception as e2:
        print("Load after fix failed with:")
        traceback.print_exc()
