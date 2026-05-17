import h5py
import json

with open('print_config_dtypes.log', 'w') as log_file:
    def log_print(*args, **kwargs):
        print(*args, file=log_file, **kwargs)

    with h5py.File('app/simple_cnn.h5', 'r') as f:
        model_config = f.attrs.get('model_config')
        if model_config:
            if hasattr(model_config, 'decode'):
                model_config = model_config.decode('utf-8')
            config = json.loads(model_config)
            
            if isinstance(config['config'], dict) and 'layers' in config['config']:
                for i, layer in enumerate(config['config']['layers']):
                    lyr_config = layer.get('config')
                    if lyr_config:
                        log_print(f"Layer {i} ({layer.get('class_name')}): dtype = {lyr_config.get('dtype')}")
