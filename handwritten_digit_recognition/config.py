"""
Configuration settings for Handwritten Digit Recognition system.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "digit_cnn_model.keras")

# ─── Image Settings ──────────────────────────────────────────────────────────
IMG_HEIGHT = 28
IMG_WIDTH = 28
IMG_CHANNELS = 1  # Grayscale

# ─── CNN Model Hyperparameters ───────────────────────────────────────────────
NUM_CLASSES = 10          # Digits 0-9
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.1

# ─── CNN Architecture ────────────────────────────────────────────────────────
CONV_FILTERS = [32, 64]          # Filters per conv layer
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)
DENSE_UNITS = 128
DROPOUT_RATE = 0.5

# ─── GUI Settings ────────────────────────────────────────────────────────────
CANVAS_SIZE = 280          # Drawing canvas size in pixels
LINE_WIDTH = 18            # Brush thickness for drawing
BG_COLOR = "black"
FG_COLOR = "white"

# ─── Prediction ──────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to display prediction
