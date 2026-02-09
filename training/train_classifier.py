#!/usr/bin/env python3
"""
Train a simple CNN classifier on the dataset folder structure:

  dataset/
    train/
      <class1>/...
      <class2>/...
    validation/
      <class1>/...
      <class2>/...

This script trains a Keras model and saves SavedModel and (optionally) ONNX/TFLite exports.

Usage:
  python training/train_classifier.py --data_dir dataset --out_dir models --img_size 128 --epochs 20

Requirements:
  pip install tensorflow tensorflow-io tf2onnx
  (tf2onnx optional; for ONNX export)

"""
import argparse
import os
from pathlib import Path
import tensorflow as tf


def build_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset', help='Path to dataset root')
    parser.add_argument('--out_dir', type=str, default='models', help='Where to save trained models')
    parser.add_argument('--img_size', type=int, default=128, help='Image size (square)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dir = data_dir / 'train'
    val_dir = data_dir / 'validation'
    if not train_dir.exists() or not val_dir.exists():
        raise SystemExit('dataset must contain train/ and validation/ subfolders with class subfolders')

    img_size = (args.img_size, args.img_size)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print('Classes:', class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    model = build_model(input_shape=(args.img_size, args.img_size, 3), num_classes=num_classes)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(out_dir / 'best_model.h5'), save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor='val_loss')
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    # Save TensorFlow SavedModel (needed for TFLite conversion) and export TFLite
    saved_model_dir = out_dir / 'saved_model'
    try:
        if hasattr(model, 'export'):
            # newer Keras/TF versions: export() writes a SavedModel
            model.export(str(saved_model_dir))
        else:
            # fallback to the tf.keras save utility (legacy)
            tf.keras.models.save_model(model, str(saved_model_dir), save_format='tf')
        print('Saved SavedModel to', saved_model_dir)
    except Exception as ex:
        print('SavedModel export failed:', ex)

    # Convert SavedModel to TFLite only
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        # default optimization; users can change or add quantization later
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        tflite_path = out_dir / 'model.tflite'
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print('Saved TFLite to', tflite_path)
    except Exception as ex:
        print('TFLite export skipped or failed:', ex)

    # Save class names for mapping predictions back to characters
    with open(out_dir / 'classes.txt', 'w', encoding='utf-8') as f:
        for c in class_names:
            f.write(c + '\n')


if __name__ == '__main__':
    main()
