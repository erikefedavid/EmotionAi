import h5py
import json

with h5py.File('app/simple_cnn.h5', 'r') as f:
    model_config = f.attrs.get('model_config')
    if model_config:
        if hasattr(model_config, 'decode'):
            model_config = model_config.decode('utf-8')
        config = json.loads(model_config)
        
        with open('model_config_full.json', 'w') as out:
            json.dump(config, out, indent=2)
        print("Wrote model_config_full.json")
