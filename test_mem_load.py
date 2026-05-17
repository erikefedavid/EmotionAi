import os
import sys
import h5py
import json
import traceback

os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow as tf
from tensorflow.keras.models import model_from_json

MODEL_PATH = 'app/simple_cnn.h5'

try:
    print("Opening file read-only...")
    with h5py.File(MODEL_PATH, 'r') as f:
        model_config = f.attrs.get('model_config')
        if model_config:
            if hasattr(model_config, 'decode'):
                model_config = model_config.decode('utf-8')
            
            config = json.loads(model_config)
            print("Loaded config successfully.")
            
            # Apply our fixes to the config dict in memory
            if isinstance(config, dict):
                config.pop('registered_name', None)
                config.pop('build_input_shape', None)
                if 'config' in config and isinstance(config['config'], dict):
                    config['config'].pop('registered_name', None)
                    config['config'].pop('build_input_shape', None)
            
            def fix_layer(layer):
                if not isinstance(layer, dict):
                    return
                if 'config' in layer and isinstance(layer['config'], dict):
                    layer['config'].pop('batch_shape', None)
                    layer['config'].pop('registered_name', None)
                    layer['config'].pop('optional', None)
                    layer['config'].pop('quantization_config', None)
                    layer['config'].pop('build_input_shape', None)
                    
                    dtype = layer['config'].get('dtype')
                    if isinstance(dtype, dict) and 'config' in dtype:
                        layer['config']['dtype'] = dtype['config'].get('name', 'float32')
                    
                    keys_to_simplify = [
                        'kernel_initializer', 'bias_initializer', 
                        'kernel_regularizer', 'bias_regularizer', 'activity_regularizer',
                        'beta_initializer', 'gamma_initializer',
                        'moving_mean_initializer', 'moving_variance_initializer'
                    ]
                    for key in keys_to_simplify:
                        val = layer['config'].get(key)
                        if isinstance(val, dict) and 'class_name' in val:
                            layer['config'][key] = {
                                'class_name': val['class_name'],
                                'config': val.get('config', {})
                            }
                
                if 'layers' in layer and isinstance(layer['layers'], list):
                    for sub in layer['layers']:
                        fix_layer(sub)
            
            fix_layer(config)
            
            print("Building model from JSON...")
            model = model_from_json(json.dumps(config))
            print("Model built successfully! Loading weights...")
            model.load_weights(MODEL_PATH)
            print("SUCCESS! Model loaded from config + weights!")
except Exception as e:
    traceback.print_exc()
