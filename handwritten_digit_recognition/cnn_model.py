"""
CNN Model Module
Defines, trains, evaluates, and manages the Convolutional Neural Network
for handwritten digit recognition.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

import config
from image_preprocessor import load_mnist_data, create_data_augmentation


def build_model(use_augmentation: bool = True) -> tf.keras.Model:
    """
    Build a CNN model for digit classification.

    Architecture:
        Input → [DataAugmentation] → Conv2D → Conv2D → MaxPool → Dropout
        → Conv2D → Conv2D → MaxPool → Dropout → Flatten → Dense → Dropout → Output

    Args:
        use_augmentation: Whether to include data augmentation layers.

    Returns:
        tf.keras.Model: Compiled CNN model.
    """
    inputs = layers.Input(
        shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
        name="input_image",
    )

    x = inputs

    # Optional data augmentation (only active during training)
    if use_augmentation:
        augmentation = create_data_augmentation()
        x = augmentation(x)

    # ── Block 1 ──────────────────────────────────────────────────────────
    x = layers.Conv2D(
        config.CONV_FILTERS[0], config.KERNEL_SIZE,
        activation="relu", padding="same", name="conv1a",
    )(x)
    x = layers.BatchNormalization(name="bn1a")(x)
    x = layers.Conv2D(
        config.CONV_FILTERS[0], config.KERNEL_SIZE,
        activation="relu", padding="same", name="conv1b",
    )(x)
    x = layers.BatchNormalization(name="bn1b")(x)
    x = layers.MaxPooling2D(config.POOL_SIZE, name="pool1")(x)
    x = layers.Dropout(0.25, name="drop1")(x)

    # ── Block 2 ──────────────────────────────────────────────────────────
    x = layers.Conv2D(
        config.CONV_FILTERS[1], config.KERNEL_SIZE,
        activation="relu", padding="same", name="conv2a",
    )(x)
    x = layers.BatchNormalization(name="bn2a")(x)
    x = layers.Conv2D(
        config.CONV_FILTERS[1], config.KERNEL_SIZE,
        activation="relu", padding="same", name="conv2b",
    )(x)
    x = layers.BatchNormalization(name="bn2b")(x)
    x = layers.MaxPooling2D(config.POOL_SIZE, name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)

    # ── Classifier Head ──────────────────────────────────────────────────
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(config.DENSE_UNITS, activation="relu", name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dropout(config.DROPOUT_RATE, name="drop_dense")(x)
    outputs = layers.Dense(config.NUM_CLASSES, activation="softmax", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="DigitCNN")

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model(model: tf.keras.Model = None, verbose: int = 1):
    """
    Train the CNN model on the MNIST dataset.

    Args:
        model: Pre-built model (builds a new one if None).
        verbose: Verbosity level for training output.

    Returns:
        tuple: (trained model, training history)
    """
    # Load data
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    # Build model if not provided
    if model is None:
        model = build_model(use_augmentation=True)

    print("\n[Model] Architecture Summary:")
    model.summary()

    # Callbacks
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # Train
    print("\n[Model] Starting training...")
    history = model.fit(
        x_train, y_train,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_split=config.VALIDATION_SPLIT,
        callbacks=cb_list,
        verbose=verbose,
    )

    # Evaluate on test set
    print("\n[Model] Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"[Model] Test Loss     : {test_loss:.4f}")
    print(f"[Model] Test Accuracy : {test_accuracy:.4f}")

    # Save model
    save_model(model)

    return model, history


def save_model(model: tf.keras.Model):
    """Save the trained model to disk."""
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    model.save(config.MODEL_PATH)
    print(f"[Model] Saved to {config.MODEL_PATH}")


def load_model() -> tf.keras.Model:
    """
    Load a trained model from disk.

    Returns:
        tf.keras.Model: Loaded model, or None if not found.
    """
    if not os.path.exists(config.MODEL_PATH):
        print(f"[Model] No saved model found at {config.MODEL_PATH}")
        return None

    model = tf.keras.models.load_model(config.MODEL_PATH)
    print(f"[Model] Loaded from {config.MODEL_PATH}")
    return model


def get_model() -> tf.keras.Model:
    """
    Get a trained model — loads from disk if available, otherwise trains a new one.

    Returns:
        tf.keras.Model: Ready-to-use model.
    """
    model = load_model()
    if model is None:
        print("[Model] No pre-trained model found. Training a new one...")
        model, _ = train_model()
    return model
