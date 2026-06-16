"""Grad-CAM visualization for the trained VGG16 RAVDESS model."""
from __future__ import annotations

from pathlib import Path
import argparse

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from config import IMG_SIZE, CLASS_NAMES


def load_image(image_path: Path):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    x = img_resized.astype("float32") / 255.0
    return img_rgb, np.expand_dims(x, axis=0)


def find_last_conv_layer(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if isinstance(layer, tf.keras.Model):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub.name
    raise ValueError("No Conv2D layer found.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    # Find layer even if nested in VGG16 model
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
        conv_model = model
    except ValueError:
        base_model = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))
        last_conv_layer = base_model.get_layer(last_conv_layer_name)
        conv_model = base_model

    # Simpler robust approach: rebuild feature extractor from nested VGG16 if present
    if conv_model is model:
        grad_model = tf.keras.models.Model(
            [model.inputs], [last_conv_layer.output, model.output]
        )
    else:
        base_model = conv_model
        conv_output_model = tf.keras.models.Model(base_model.inputs, last_conv_layer.output)
        classifier_input = model.input
        base_output = model.get_layer(base_model.name)(classifier_input)
        classifier_output = base_output
        start = False
        for layer in model.layers:
            if start:
                classifier_output = layer(classifier_output)
            if layer.name == base_model.name:
                start = True
        grad_model = tf.keras.models.Model(model.inputs, [conv_output_model(model.input), classifier_output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), predictions.numpy()[0]


def save_gradcam_figure(model_path: Path, image_path: Path, true_label: str, output_path: Path):
    model = tf.keras.models.load_model(str(model_path))
    original_rgb, img_array = load_image(image_path)
    heatmap, proba = make_gradcam_heatmap(img_array, model)

    pred_idx = int(np.argmax(proba))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(proba[pred_idx] * 100)

    resized_original = cv2.resize(original_rgb, (IMG_SIZE, IMG_SIZE))
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.55 * resized_original + 0.45 * colored_heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"Grad-CAM Visualization - {true_label.upper()}", fontsize=16, fontweight="bold")

    axes[0].imshow(resized_original)
    axes[0].set_title(f"(a) Original Frame\nTrue: {true_label}", fontweight="bold")
    axes[0].axis("off")

    im = axes[1].imshow(heatmap_resized, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title(f"(b) Grad-CAM Heatmap\nPred: {pred_label} ({confidence:.1f}%)", fontweight="bold")
    axes[1].axis("off")
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Attention Intensity")

    axes[2].imshow(overlay)
    axes[2].set_title("(c) Attention Overlay", fontweight="bold")
    axes[2].axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Grad-CAM figure: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path, help="Path to trained .keras model")
    parser.add_argument("--image", required=True, type=Path, help="Path to input frame image")
    parser.add_argument("--true_label", required=True, type=str, help="Ground-truth emotion label")
    parser.add_argument("--output", required=True, type=Path, help="Output PNG path")
    args = parser.parse_args()

    save_gradcam_figure(args.model, args.image, args.true_label, args.output)


if __name__ == "__main__":
    main()
