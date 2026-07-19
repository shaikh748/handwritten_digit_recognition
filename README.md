# Handwritten Digit Recognition

An end-to-end Computer Vision and Deep Learning application designed to recognize handwritten digits. This project features a Convolutional Neural Network (CNN) built with TensorFlow, coupled with an interactive Graphical User Interface (GUI) that allows users to draw and classify digits in real-time.

---

## 🚀 Features

* **CNN Deep Learning Model:** Uses a robust Convolutional Neural Network architecture optimized for grid-like image pixel processing.
* **Interactive GUI Application:** Features a built-in canvas workspace where users can sketch digits directly to receive instant prediction models.
* **Decoupled Pipeline Structure:** Maintains modularity by separating configuration definitions, image pre-processing steps, training mechanics, and real-time prediction scripts.

---

## 📁 Repository Structure

The code is organized into separate modules to handle each phase of the machine learning and interface workflow:

* `cnn_model.py` – Defines the layers and architecture of the Convolutional Neural Network.
* `config.py` – Manages hyperparameter constants, paths, and environment settings.
* `image_preprocessor.py` – Normalizes, resizes, and prepares raw image matrices for inference.
* `gui_app.py` – Builds the interactive desktop user interface canvas window.
* `predict.py` – Implements inference logic to load saved neural weights and process predictions.
* `main.py` – The main entry point to run and orchestrate the application pipeline.
* `models/` – Directory dedicated to storing trained model check-points and serialization artifacts.

---

## 🛠️ Installation & Dependencies

Make sure Python is installed on your local system before setting up the repository environment.

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/shaikh748/handwritten_digit_recognition.git](https://github.com/shaikh748/handwritten_digit_recognition.git)
   cd handwritten_digit_recognition
