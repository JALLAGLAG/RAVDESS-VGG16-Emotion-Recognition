"""Actor-independent 5-fold cross-validation for RAVDESS VGG16.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from config import (
    FRAMES_DIR, MODELS_DIR, FIGURES_DIR, CLASS_NAMES, BATCH_SIZE, EPOCHS,
    IMG_SIZE, N_FOLDS, RANDOM_SEED, BEST_MODEL_NAME
)
from model import build_vgg16_model
from utils import extract_actor, group_by_actor, video_confidence_vote, print_metrics


def collect_frame_folders(frames_dir: Path):
    video_dirs, labels, actors = [], [], []
    for emotion in CLASS_NAMES:
        emotion_dir = frames_dir / emotion
        if not emotion_dir.exists():
            continue
        for video_dir in sorted([p for p in emotion_dir.iterdir() if p.is_dir()]):
            video_dirs.append(video_dir)
            labels.append(CLASS_NAMES.index(emotion))
            actors.append(extract_actor(video_dir.name))
    return np.array(video_dirs), np.array(labels), np.array(actors)


def make_frame_dataset(video_dirs, labels, training=True):
    image_paths, image_labels, video_index = [], [], []
    for i, (video_dir, label) in enumerate(zip(video_dirs, labels)):
        frames = sorted(list(Path(video_dir).glob("*.jpg")))
        for frame in frames:
            image_paths.append(str(frame))
            image_labels.append(int(label))
            video_index.append(i)

    def load_image(path, label, vid_idx):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = tf.cast(img, tf.float32) / 255.0
        if training:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.08)
            img = tf.image.random_contrast(img, 0.9, 1.1)
        return img, label, vid_idx

    ds = tf.data.Dataset.from_tensor_slices((image_paths, image_labels, video_index))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(10000, seed=RANDOM_SEED)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds, np.array(image_labels), np.array(video_index)


def train_one_fold(fold_id, train_dirs, train_labels, test_dirs, test_labels):
    print("\n" + "=" * 80)
    print(f"FOLD {fold_id}")
    print("=" * 80)

    train_ds, train_frame_labels, _ = make_frame_dataset(train_dirs, train_labels, training=True)
    test_ds, test_frame_labels, test_video_index = make_frame_dataset(test_dirs, test_labels, training=False)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(CLASS_NAMES)),
        y=train_frame_labels,
    )
    class_weight = {i: float(w) for i, w in enumerate(weights)}

    model = build_vgg16_model()
    fold_model_path = MODELS_DIR / f"fold_{fold_id}_{BEST_MODEL_NAME}"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(fold_model_path), monitor="val_accuracy", save_best_only=True, mode="max"),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]

    train_ds_model = train_ds.map(lambda x, y, z: (x, y))
    test_ds_model = test_ds.map(lambda x, y, z: (x, y))

    model.fit(
        train_ds_model,
        validation_data=test_ds_model,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    probabilities = model.predict(test_ds_model, verbose=1)
    frame_preds = np.argmax(probabilities, axis=1)
    frame_acc = accuracy_score(test_frame_labels, frame_preds)

    # Video-level confidence voting
    video_preds, video_true = [], []
    for local_idx in range(len(test_dirs)):
        mask = test_video_index == local_idx
        if not np.any(mask):
            continue
        pred = video_confidence_vote(probabilities[mask])
        video_preds.append(pred)
        video_true.append(int(test_labels[local_idx]))

    video_acc = accuracy_score(video_true, video_preds)
    print_metrics(video_true, video_preds, title=f"Fold {fold_id} video-level results")

    return {
        "fold": fold_id,
        "frame_accuracy": frame_acc,
        "video_accuracy": video_acc,
        "n_train_videos": len(train_dirs),
        "n_test_videos": len(test_dirs),
    }


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    video_dirs, labels, actors = collect_frame_folders(FRAMES_DIR)
    unique_actors = np.array(sorted(set(actors)))
    print(f"Videos: {len(video_dirs)} | Actors: {len(unique_actors)} | Actors={unique_actors.tolist()}")

    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    results = []

    for fold_id, (train_actor_idx, test_actor_idx) in enumerate(kfold.split(unique_actors), start=1):
        train_actors = set(unique_actors[train_actor_idx])
        test_actors = set(unique_actors[test_actor_idx])

        train_mask = np.array([a in train_actors for a in actors])
        test_mask = np.array([a in test_actors for a in actors])

        assert set(actors[train_mask]).isdisjoint(set(actors[test_mask])), "Actor leakage detected!"

        result = train_one_fold(
            fold_id,
            video_dirs[train_mask], labels[train_mask],
            video_dirs[test_mask], labels[test_mask],
        )
        result["train_actors"] = sorted(list(train_actors))
        result["test_actors"] = sorted(list(test_actors))
        results.append(result)

    df = pd.DataFrame(results)
    out_csv = FIGURES_DIR / "actor_independent_5fold_results.csv"
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print("5-FOLD SUMMARY")
    print("=" * 80)
    print(df[["fold", "frame_accuracy", "video_accuracy"]])
    print(f"Frame mean ± std: {df.frame_accuracy.mean():.4f} ± {df.frame_accuracy.std():.4f}")
    print(f"Video mean ± std: {df.video_accuracy.mean():.4f} ± {df.video_accuracy.std():.4f}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
