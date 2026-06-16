"""Utility functions for RAVDESS filename parsing, metrics, and voting."""
from __future__ import annotations

import os
import re
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import EMOTION_MAP, CLASS_NAMES


def find_video_files(data_dir: Path) -> List[Path]:
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    return sorted([p for p in Path(data_dir).rglob("*") if p.suffix.lower() in exts])


def extract_actor(video_path: Path) -> int:
    text = str(video_path)
    match = re.search(r"Actor[_-](\d+)", text)
    if match:
        return int(match.group(1))

    stem = Path(video_path).stem
    parts = stem.split("-")
    if len(parts) >= 7:
        return int(parts[-1])
    raise ValueError(f"Cannot extract actor ID from: {video_path}")


def extract_emotion(video_path: Path) -> str:
    stem = Path(video_path).stem
    parts = stem.split("-")
    if len(parts) >= 3 and parts[2] in EMOTION_MAP:
        return EMOTION_MAP[parts[2]]

    # fallback: emotion appears in folder name
    lower_parts = [part.lower() for part in Path(video_path).parts]
    for emotion in CLASS_NAMES:
        if emotion in lower_parts:
            return emotion
    raise ValueError(f"Cannot extract emotion label from: {video_path}")


def group_by_actor(video_paths: List[Path]) -> Dict[int, List[Path]]:
    actor_to_videos: Dict[int, List[Path]] = {}
    for video in video_paths:
        actor = extract_actor(video)
        actor_to_videos.setdefault(actor, []).append(video)
    return actor_to_videos


def video_majority_vote(frame_predictions: List[int]) -> int:
    return Counter(frame_predictions).most_common(1)[0][0]


def video_confidence_vote(frame_probabilities: np.ndarray) -> int:
    return int(np.argmax(np.sum(frame_probabilities, axis=0)))


def print_metrics(y_true, y_pred, title: str = "Results") -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
