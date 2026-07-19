# ✏️ Handwritten Digit Recognition

A CNN-based system that recognizes handwritten digits (0–9) using TensorFlow and the MNIST dataset. Includes a Tkinter GUI for drawing digits and getting real-time predictions.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Model Architecture](#model-architecture)
- [Learning Outcomes](#learning-outcomes)

---

## 🔍 Overview

| Aspect | Detail |
|---|---|
| **What it is** | Recognizes handwritten digits (0–9) |
| **Problem solved** | Automates manual digit recognition |
| **Working** | CNN-based image classification |
| **Core modules** | Image preprocessor, CNN model |
| **Tech stack** | Python, TensorFlow, Tkinter |
| **Learning outcome** | Deep learning basics |

---

## ✨ Features

- **CNN Model** — Multi-layer convolutional neural network with batch normalization and dropout
- **MNIST Training** — Automatic training on the MNIST dataset (60,000 training + 10,000 test images)
- **Data Augmentation** — Random rotation, zoom, and translation for better generalization
- **Drawing GUI** — Tkinter canvas for freehand digit drawing
- **Real-time Prediction** — Instant digit recognition with confidence scores
- **Probability Display** — Visual progress bars showing probabilities for all 10 digits
- **Image File Support** — Load and predict from external image files
- **CLI Interface** — Command-line options for training, evaluation, and prediction

---

## 📁 Project Structure

```
handwritten_digit_recognition/
├── config.py               # Configuration and hyperparameters
├── image_preprocessor.py   # Image loading, preprocessing, augmentation
├── cnn_model.py            # CNN architecture, training, saving/loading
├── predict.py              # Prediction engine and result formatting
├── gui_app.py              # Tkinter GUI application
├── main.py                 # Entry point with CLI argument parsing
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── models/                 # Saved trained models
    └── digit_cnn_model.keras
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.8+** | Core programming language |
| **TensorFlow / Keras** | Deep learning framework for CNN |
| **NumPy** | Numerical operations and array manipulation |
| **Pillow (PIL)** | Image processing and manipulation |
| **Tkinter** | GUI framework (built into Python) |

---

## 📦 Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd handwritten_digit_recognition
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/Mac
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### Launch the GUI (default)
```bash
python main.py
```
Draw a digit on the canvas and click **Predict** to see the result.

### Train the model
```bash
python main.py --train
```
Trains the CNN on the MNIST dataset and saves the model to `models/`.

### Evaluate the model
```bash
python main.py --evaluate
```
Runs evaluation on the MNIST test set and prints per-digit accuracy.

### Predict from an image file
```bash
python main.py --predict path/to/digit.png
```

### View model architecture
```bash
python main.py --summary
```

---

## ⚙️ How It Works

### 1. Image Preprocessing (`image_preprocessor.py`)
- Loads the MNIST dataset (auto-downloaded by TensorFlow)
- Normalizes pixel values to [0, 1]
- Reshapes images to (28, 28, 1) for the CNN
- For canvas/file input: converts to grayscale, resizes, centres the digit, and normalizes

### 2. CNN Model (`cnn_model.py`)
- Builds a multi-layer CNN with configurable architecture
- Trains with data augmentation, early stopping, and learning rate reduction
- Saves/loads models in Keras format

### 3. Prediction (`predict.py`)
- Wraps the model in a `DigitPredictor` class
- Accepts PIL images, file paths, or numpy arrays
- Returns digit, confidence, and full probability distribution

### 4. GUI (`gui_app.py`)
- Drawing canvas with mouse input
- Real-time probability bar chart for all 10 digits
- Load external images for prediction
- Clear and reset functionality

---

## 🧠 Model Architecture

```
Input (28×28×1)
    ↓
[Data Augmentation] (rotation, zoom, translation — training only)
    ↓
Conv2D(32, 3×3) → BatchNorm → Conv2D(32, 3×3) → BatchNorm → MaxPool(2×2) → Dropout(0.25)
    ↓
Conv2D(64, 3×3) → BatchNorm → Conv2D(64, 3×3) → BatchNorm → MaxPool(2×2) → Dropout(0.25)
    ↓
Flatten → Dense(128) → BatchNorm → Dropout(0.5)
    ↓
Dense(10, softmax) → Output
```

**Expected accuracy:** ~99%+ on the MNIST test set.

---

## 📚 Learning Outcomes

- **Deep Learning Fundamentals** — Understanding CNNs, convolution, pooling, and fully connected layers
- **TensorFlow / Keras** — Building, training, and deploying neural networks
- **Image Preprocessing** — Normalization, resizing, centering, and augmentation techniques
- **Model Management** — Saving, loading, and evaluating trained models
- **GUI Development** — Building interactive applications with Tkinter
- **Software Architecture** — Modular design with separation of concerns
