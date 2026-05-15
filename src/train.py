import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.utils import class_weight
from pathlib import Path
from src.utils import get_root_dir, plot_training_history

def build_resnet50(num_classes=6):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
    
    # Custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

def train():
    root_dir = get_root_dir()
    train_dir = root_dir / 'data' / 'train'
    val_dir = root_dir / 'data' / 'val'
    
    batch_size = 32
    num_classes = 6
    
    # Image generators with ImageNet normalization
    # Normalization as per PRD: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # We can use tf.keras.applications.resnet50.preprocess_input
    from tensorflow.keras.applications.resnet50 import preprocess_input
    
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    # Compute class weights
    classes = train_generator.classes
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(classes), y=classes)
    class_weights = dict(enumerate(weights))
    
    model, base_model = build_resnet50(num_classes)
    
    # Phase 1: Feature Extraction (Freeze base layers)
    for layer in base_model.layers:
        layer.trainable = False
        
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    
    checkpoint_path = root_dir / 'models' / 'resnet50_fer.h5'
    log_path = root_dir / 'results' / 'logs' / 'training_log.csv'
    
    callbacks = [
        ModelCheckpoint(str(checkpoint_path), monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=8, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
        CSVLogger(str(log_path), append=True)
    ]
    
    print("Starting Phase 1: Training custom head...")
    history_p1 = model.fit(
        train_generator,
        epochs=30,
        validation_data=val_generator,
        class_weight=class_weights,
        callbacks=callbacks
    )
    
    # Phase 2: Fine-tuning (Unfreeze last 2 blocks)
    # ResNet50 has 5 stages. Last stage starts around layer 143 (conv5_block1_1_conv)
    for layer in base_model.layers[-15:]: # Roughly last 2-3 blocks
        layer.trainable = True
        
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Starting Phase 2: Fine-tuning last layers...")
    history_p2 = model.fit(
        train_generator,
        epochs=20,
        initial_epoch=history_p1.epoch[-1] + 1 if history_p1.epoch else 30,
        validation_data=val_generator,
        class_weight=class_weights,
        callbacks=callbacks
    )
    
    # Save final curves
    plot_training_history(history_p2, root_dir / 'results' / 'figures' / 'training_curves.png')
    print("Training complete.")

if __name__ == '__main__':
    train()
