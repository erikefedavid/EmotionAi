import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from src.utils import get_root_dir

def build_vgg16(num_classes=6):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=predictions)

def build_mobilenetv2(num_classes=6):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=predictions)

def build_custom_cnn(num_classes=6):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def train_baseline(model_name, build_func):
    root_dir = get_root_dir()
    print(f"Training baseline: {model_name}...")
    
    train_dir = root_dir / 'data' / 'train'
    val_dir = root_dir / 'data' / 'val'
    
    datagen = ImageDataGenerator(rescale=1./255) # Simple rescaling for baselines
    
    train_gen = datagen.flow_from_directory(train_dir, target_size=(48, 48), batch_size=32, class_mode='categorical')
    val_gen = datagen.flow_from_directory(val_dir, target_size=(48, 48), batch_size=32, class_mode='categorical')
    
    model = build_func()
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    
    checkpoint = ModelCheckpoint(str(root_dir / 'models' / f'{model_name}_fer.h5'), save_best_only=True)
    csv_logger = CSVLogger(str(root_dir / 'results' / 'logs' / f'{model_name}_log.csv'))
    
    model.fit(train_gen, epochs=20, validation_data=val_gen, callbacks=[checkpoint, csv_logger])

if __name__ == '__main__':
    # We will train these on Colab as well
    # train_baseline('vgg16', build_vgg16)
    # train_baseline('mobilenetv2', build_mobilenetv2)
    # train_baseline('custom_cnn', build_custom_cnn)
    pass
