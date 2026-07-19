"""
Main Entry Point
Handwritten Digit Recognition System

Usage:
    python main.py              — Launch the GUI application
    python main.py --train      — Train the CNN model
    python main.py --evaluate   — Evaluate the trained model
    python main.py --predict <image_path>  — Predict digit from an image file
"""

import argparse
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


def main():
    parser = argparse.ArgumentParser(
        description="Handwritten Digit Recognition using CNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    Launch the drawing GUI
  python main.py --train            Train the CNN model on MNIST
  python main.py --evaluate         Evaluate model on test set
  python main.py --predict digit.png  Predict digit from image file
        """,
    )

    parser.add_argument(
        "--train", action="store_true",
        help="Train the CNN model on the MNIST dataset",
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Evaluate the trained model on the MNIST test set",
    )
    parser.add_argument(
        "--predict", type=str, metavar="IMAGE",
        help="Predict the digit in the given image file",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print the model architecture summary",
    )

    args = parser.parse_args()

    # ── Train ────────────────────────────────────────────────────────────
    if args.train:
        _train()
        return

    # ── Evaluate ─────────────────────────────────────────────────────────
    if args.evaluate:
        _evaluate()
        return

    # ── Predict from file ────────────────────────────────────────────────
    if args.predict:
        _predict_file(args.predict)
        return

    # ── Model summary ────────────────────────────────────────────────────
    if args.summary:
        _summary()
        return

    # ── Default: launch GUI ──────────────────────────────────────────────
    _launch_gui()


def _train():
    """Train the CNN model."""
    from cnn_model import train_model

    print("=" * 60)
    print("  Handwritten Digit Recognition — Training")
    print("=" * 60)
    model, history = train_model()

    final_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    print(f"\n[Done] Final training accuracy   : {final_acc:.4f}")
    print(f"[Done] Final validation accuracy : {final_val_acc:.4f}")
    print(f"[Done] Model saved to: {config.MODEL_PATH}")


def _evaluate():
    """Evaluate the trained model on the MNIST test set."""
    import tensorflow as tf
    from cnn_model import load_model
    from image_preprocessor import load_mnist_data

    print("=" * 60)
    print("  Handwritten Digit Recognition — Evaluation")
    print("=" * 60)

    model = load_model()
    if model is None:
        print("[Error] No trained model found. Run with --train first.")
        sys.exit(1)

    (_, _), (x_test, y_test) = load_mnist_data()
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)

    print(f"\n[Result] Test Loss     : {loss:.4f}")
    print(f"[Result] Test Accuracy : {accuracy:.4f}")

    # Per-class accuracy
    import numpy as np
    predictions = model.predict(x_test, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    print("\nPer-digit accuracy:")
    for digit in range(config.NUM_CLASSES):
        mask = true_classes == digit
        digit_acc = np.mean(pred_classes[mask] == digit)
        count = np.sum(mask)
        print(f"  Digit {digit}: {digit_acc:.4f} ({count} samples)")


def _predict_file(image_path: str):
    """Predict digit from an image file."""
    from predict import DigitPredictor, format_prediction

    if not os.path.exists(image_path):
        print(f"[Error] File not found: {image_path}")
        sys.exit(1)

    print("=" * 60)
    print("  Handwritten Digit Recognition — Prediction")
    print("=" * 60)
    print(f"  Image: {image_path}")
    print("=" * 60)

    predictor = DigitPredictor()
    result = predictor.predict_from_file(image_path)
    print(f"\n{format_prediction(result)}")


def _summary():
    """Print model architecture summary."""
    from cnn_model import build_model

    print("=" * 60)
    print("  CNN Model Architecture")
    print("=" * 60)
    model = build_model(use_augmentation=False)
    model.summary()


def _launch_gui():
    """Launch the GUI application."""
    print("=" * 60)
    print("  Handwritten Digit Recognition")
    print("  Launching GUI...")
    print("=" * 60)

    from gui_app import run_gui
    run_gui()


if __name__ == "__main__":
    main()
