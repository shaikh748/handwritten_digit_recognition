"""
Image Preprocessor Module
Handles loading, preprocessing, and augmenting images for the CNN model.
"""

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

import config


def load_mnist_data():
    """
    Load and preprocess the MNIST dataset.

    Returns:
        tuple: (x_train, y_train), (x_test, y_test) — normalised and reshaped.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Reshape to (samples, height, width, channels)
    x_train = x_train.reshape(-1, config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)
    x_test = x_test.reshape(-1, config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)

    # Normalise pixel values to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # One-hot encode labels
    y_train = tf.keras.utils.to_categorical(y_train, config.NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, config.NUM_CLASSES)

    print(f"[Preprocessor] Training samples : {x_train.shape[0]}")
    print(f"[Preprocessor] Test samples     : {x_test.shape[0]}")
    print(f"[Preprocessor] Image shape      : {x_train.shape[1:]}")

    return (x_train, y_train), (x_test, y_test)


def create_data_augmentation():
    """
    Create a data augmentation pipeline using Keras layers.

    Returns:
        tf.keras.Sequential: Data augmentation model.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ], name="data_augmentation")


def preprocess_canvas_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess a PIL image drawn on the GUI canvas for prediction.

    Steps:
        1. Convert to grayscale.
        2. Resize to 28×28.
        3. Invert if background is light.
        4. Centre the digit using centre-of-mass.
        5. Normalise to [0, 1].

    Args:
        image: PIL Image from the drawing canvas.

    Returns:
        np.ndarray: Preprocessed image array of shape (1, 28, 28, 1).
    """
    # Convert to grayscale
    img = image.convert("L")

    # Resize to 28×28 with anti-aliasing
    img = img.resize((config.IMG_WIDTH, config.IMG_HEIGHT), Image.LANCZOS)

    # Convert to numpy array
    img_array = np.array(img, dtype="float32")

    # Invert if the background is light (MNIST has white digits on black bg)
    if np.mean(img_array) > 127:
        img_array = 255.0 - img_array

    # Centre the digit using centre of mass
    img_array = _centre_digit(img_array)

    # Normalise to [0, 1]
    img_array = img_array / 255.0

    # Reshape for model input: (1, 28, 28, 1)
    img_array = img_array.reshape(1, config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)

    return img_array


def preprocess_image_file(file_path: str) -> np.ndarray:
    """
    Preprocess an image file for prediction.

    Args:
        file_path: Path to the image file.

    Returns:
        np.ndarray: Preprocessed image array of shape (1, 28, 28, 1).
    """
    img = Image.open(file_path)
    return preprocess_canvas_image(img)


def _centre_digit(img_array: np.ndarray) -> np.ndarray:
    """
    Centre the digit in the 28×28 image using centre of mass.

    Args:
        img_array: 2D numpy array (28×28).

    Returns:
        np.ndarray: Centred image array.
    """
    # Threshold to find digit pixels
    threshold = 50
    rows = np.any(img_array > threshold, axis=1)
    cols = np.any(img_array > threshold, axis=0)

    if not np.any(rows) or not np.any(cols):
        return img_array  # Empty image, return as-is

    # Find bounding box
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop to bounding box
    cropped = img_array[rmin:rmax + 1, cmin:cmax + 1]

    # Resize cropped digit to 20×20 (MNIST convention: 20×20 digit in 28×28 frame)
    cropped_img = Image.fromarray(cropped.astype("uint8"))
    cropped_img = cropped_img.resize((20, 20), Image.LANCZOS)

    # Place in centre of 28×28 canvas
    result = np.zeros((28, 28), dtype="float32")
    result[4:24, 4:24] = np.array(cropped_img, dtype="float32")

    return result
