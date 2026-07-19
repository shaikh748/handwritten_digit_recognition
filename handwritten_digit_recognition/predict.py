"""
Prediction Module
Handles digit prediction from preprocessed images using the trained CNN model.
"""

import numpy as np
from PIL import Image

import config
from cnn_model import get_model
from image_preprocessor import preprocess_canvas_image, preprocess_image_file


class DigitPredictor:
    """Encapsulates the trained CNN model for digit prediction."""

    def __init__(self):
        """Initialise the predictor by loading the trained model."""
        self.model = get_model()

    def predict_from_image(self, image: Image.Image) -> dict:
        """
        Predict the digit from a PIL Image (e.g., from the drawing canvas).

        Args:
            image: PIL Image of the handwritten digit.

        Returns:
            dict with keys:
                - 'digit': predicted digit (int)
                - 'confidence': prediction confidence (float)
                - 'probabilities': array of probabilities for each digit
        """
        processed = preprocess_canvas_image(image)
        return self._predict(processed)

    def predict_from_file(self, file_path: str) -> dict:
        """
        Predict the digit from an image file.

        Args:
            file_path: Path to the image file.

        Returns:
            dict with prediction results.
        """
        processed = preprocess_image_file(file_path)
        return self._predict(processed)

    def predict_from_array(self, img_array: np.ndarray) -> dict:
        """
        Predict the digit from a preprocessed numpy array.

        Args:
            img_array: Preprocessed image array of shape (1, 28, 28, 1).

        Returns:
            dict with prediction results.
        """
        return self._predict(img_array)

    def _predict(self, processed_image: np.ndarray) -> dict:
        """
        Run prediction on a preprocessed image.

        Args:
            processed_image: numpy array of shape (1, 28, 28, 1).

        Returns:
            dict with prediction results.
        """
        probabilities = self.model.predict(processed_image, verbose=0)[0]
        digit = int(np.argmax(probabilities))
        confidence = float(probabilities[digit])

        return {
            "digit": digit,
            "confidence": confidence,
            "probabilities": probabilities,
        }

    def predict_batch(self, images: list) -> list:
        """
        Predict digits for a batch of PIL Images.

        Args:
            images: List of PIL Images.

        Returns:
            List of prediction result dicts.
        """
        results = []
        for img in images:
            results.append(self.predict_from_image(img))
        return results


def format_prediction(result: dict) -> str:
    """
    Format a prediction result as a human-readable string.

    Args:
        result: Prediction result dict.

    Returns:
        Formatted string.
    """
    digit = result["digit"]
    confidence = result["confidence"]

    if confidence < config.CONFIDENCE_THRESHOLD:
        return f"Uncertain — best guess: {digit} ({confidence:.1%})"

    lines = [f"Predicted Digit: {digit}", f"Confidence: {confidence:.1%}", ""]

    # Top-3 predictions
    probs = result["probabilities"]
    top3_idx = np.argsort(probs)[::-1][:3]
    lines.append("Top 3 Predictions:")
    for rank, idx in enumerate(top3_idx, 1):
        lines.append(f"  {rank}. Digit {idx}: {probs[idx]:.1%}")

    return "\n".join(lines)
