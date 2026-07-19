"""
GUI Application Module
Provides a Tkinter-based graphical interface for drawing digits and
getting real-time predictions from the CNN model.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk
import numpy as np

import config
from predict import DigitPredictor, format_prediction


class DigitRecognitionApp:
    """Main GUI application for handwritten digit recognition."""

    def __init__(self, root: tk.Tk):
        """
        Initialise the application.

        Args:
            root: Tkinter root window.
        """
        self.root = root
        self.root.title("Handwritten Digit Recognition")
        self.root.resizable(False, False)

        # Drawing state
        self.last_x = None
        self.last_y = None

        # PIL image for capturing the drawing (used for prediction)
        self.pil_image = Image.new("L", (config.CANVAS_SIZE, config.CANVAS_SIZE), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)

        # Load predictor (this may trigger model training on first run)
        self._show_loading()
        self.root.update()
        self.predictor = DigitPredictor()
        self._hide_loading()

        # Build UI
        self._build_ui()

    # ── UI Construction ──────────────────────────────────────────────────

    def _show_loading(self):
        """Show a loading message while the model loads."""
        self.loading_label = tk.Label(
            self.root,
            text="Loading model... (first run may take a minute to train)",
            font=("Helvetica", 12),
            pady=20, padx=20,
        )
        self.loading_label.pack()

    def _hide_loading(self):
        """Remove the loading message."""
        self.loading_label.destroy()

    def _build_ui(self):
        """Construct the full user interface."""
        # ── Main Frame ───────────────────────────────────────────────────
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ── Title ────────────────────────────────────────────────────────
        title = ttk.Label(
            main_frame,
            text="✏️ Draw a Digit (0-9)",
            font=("Helvetica", 16, "bold"),
        )
        title.pack(pady=(0, 10))

        # ── Content Frame (canvas + results side by side) ────────────────
        content_frame = ttk.Frame(main_frame)
        content_frame.pack()

        # ── Canvas ───────────────────────────────────────────────────────
        canvas_frame = ttk.LabelFrame(content_frame, text="Drawing Canvas", padding=5)
        canvas_frame.pack(side=tk.LEFT, padx=(0, 10))

        self.canvas = tk.Canvas(
            canvas_frame,
            width=config.CANVAS_SIZE,
            height=config.CANVAS_SIZE,
            bg=config.BG_COLOR,
            cursor="crosshair",
        )
        self.canvas.pack()

        # Bind mouse events
        self.canvas.bind("<Button-1>", self._start_draw)
        self.canvas.bind("<B1-Motion>", self._draw)
        self.canvas.bind("<ButtonRelease-1>", self._stop_draw)

        # ── Results Panel ────────────────────────────────────────────────
        results_frame = ttk.LabelFrame(content_frame, text="Prediction", padding=10)
        results_frame.pack(side=tk.LEFT, fill=tk.BOTH)

        # Large digit display
        self.digit_label = ttk.Label(
            results_frame,
            text="?",
            font=("Helvetica", 72, "bold"),
            anchor=tk.CENTER,
        )
        self.digit_label.pack(pady=(0, 5))

        # Confidence
        self.confidence_label = ttk.Label(
            results_frame,
            text="Draw a digit to start",
            font=("Helvetica", 11),
            anchor=tk.CENTER,
        )
        self.confidence_label.pack(pady=(0, 10))

        # Probability bars
        self.prob_frame = ttk.Frame(results_frame)
        self.prob_frame.pack(fill=tk.X)

        self.prob_bars = {}
        self.prob_labels = {}
        for i in range(config.NUM_CLASSES):
            row = ttk.Frame(self.prob_frame)
            row.pack(fill=tk.X, pady=1)

            lbl = ttk.Label(row, text=f"{i}:", width=3, font=("Courier", 10))
            lbl.pack(side=tk.LEFT)

            bar = ttk.Progressbar(row, length=150, mode="determinate", maximum=100)
            bar.pack(side=tk.LEFT, padx=(2, 5))

            pct = ttk.Label(row, text="0%", width=6, font=("Courier", 9))
            pct.pack(side=tk.LEFT)

            self.prob_bars[i] = bar
            self.prob_labels[i] = pct

        # ── Buttons ──────────────────────────────────────────────────────
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="🔍 Predict", command=self._predict).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(btn_frame, text="🗑️ Clear", command=self._clear_canvas).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(btn_frame, text="📂 Load Image", command=self._load_image).pack(
            side=tk.LEFT, padx=5
        )

        # ── Status Bar ───────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="Ready — draw a digit and click Predict")
        status_bar = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=("Helvetica", 9),
        )
        status_bar.pack(fill=tk.X, pady=(5, 0))

    # ── Drawing Handlers ─────────────────────────────────────────────────

    def _start_draw(self, event):
        """Begin a new stroke."""
        self.last_x = event.x
        self.last_y = event.y

    def _draw(self, event):
        """Draw on both the Tk canvas and the PIL image."""
        if self.last_x is not None and self.last_y is not None:
            # Draw on Tk canvas
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                fill=config.FG_COLOR,
                width=config.LINE_WIDTH,
                capstyle=tk.ROUND,
                smooth=True,
            )
            # Draw on PIL image
            self.pil_draw.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill=255,
                width=config.LINE_WIDTH,
            )

        self.last_x = event.x
        self.last_y = event.y

    def _stop_draw(self, event):
        """End the current stroke."""
        self.last_x = None
        self.last_y = None

    # ── Actions ──────────────────────────────────────────────────────────

    def _predict(self):
        """Run prediction on the current canvas drawing."""
        # Check if canvas is empty
        img_array = np.array(self.pil_image)
        if np.max(img_array) < 10:
            self.status_var.set("Canvas is empty — please draw a digit first")
            return

        self.status_var.set("Predicting...")
        self.root.update()

        result = self.predictor.predict_from_image(self.pil_image)
        self._display_result(result)

        self.status_var.set(
            f"Prediction: {result['digit']} (confidence: {result['confidence']:.1%})"
        )

    def _display_result(self, result: dict):
        """Update the UI with prediction results."""
        digit = result["digit"]
        confidence = result["confidence"]
        probs = result["probabilities"]

        # Update digit display
        self.digit_label.config(text=str(digit))

        # Update confidence text
        if confidence >= config.CONFIDENCE_THRESHOLD:
            self.confidence_label.config(text=f"Confidence: {confidence:.1%}")
        else:
            self.confidence_label.config(text=f"Uncertain: {confidence:.1%}")

        # Update probability bars
        for i in range(config.NUM_CLASSES):
            pct = probs[i] * 100
            self.prob_bars[i]["value"] = pct
            self.prob_labels[i].config(text=f"{pct:.1f}%")

    def _clear_canvas(self):
        """Clear the drawing canvas and reset results."""
        self.canvas.delete("all")
        self.pil_image = Image.new("L", (config.CANVAS_SIZE, config.CANVAS_SIZE), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)

        self.digit_label.config(text="?")
        self.confidence_label.config(text="Draw a digit to start")

        for i in range(config.NUM_CLASSES):
            self.prob_bars[i]["value"] = 0
            self.prob_labels[i].config(text="0%")

        self.status_var.set("Canvas cleared — ready for a new digit")

    def _load_image(self):
        """Load an image file for prediction."""
        file_path = filedialog.askopenfilename(
            title="Select Digit Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        try:
            self.status_var.set(f"Loading {file_path}...")
            self.root.update()

            result = self.predictor.predict_from_file(file_path)
            self._display_result(result)

            # Show the loaded image on canvas
            img = Image.open(file_path).convert("L").resize(
                (config.CANVAS_SIZE, config.CANVAS_SIZE), Image.LANCZOS
            )
            self.pil_image = img
            self.pil_draw = ImageDraw.Draw(self.pil_image)

            # Display on canvas
            self._photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

            self.status_var.set(
                f"Loaded: {file_path} — Prediction: {result['digit']} "
                f"({result['confidence']:.1%})"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")
            self.status_var.set("Error loading image")


def run_gui():
    """Launch the GUI application."""
    root = tk.Tk()
    app = DigitRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
