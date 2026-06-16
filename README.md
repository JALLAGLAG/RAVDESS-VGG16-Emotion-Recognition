# Two-Stage VGG16 Emotion Recognition on RAVDESS

This repository contains the implementation used in the paper:

**A Two-Stage Deep Learning Approach for Facial Emotion Recognition in RAVDESS Videos**

The repository includes the essential scripts required for reproducibility:

1. Frame extraction from RAVDESS videos.
2. Actor-independent 5-fold cross-validation.
3. Fine-tuned VGG16 frame-level classification.
4. Video-level aggregation using confidence-based majority voting.
5. Grad-CAM visualization.

## Repository structure

```text
ravdess_vgg16_code/
├── extract_frames.py
├── model.py
├── train_5fold.py
├── gradcam.py
├── utils.py
├── requirements.txt
└── README.md
```

## Dataset

The experiments were conducted on the video-only speech subset of the RAVDESS dataset.

The dataset is not redistributed in this repository and can be obtained from its official source.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Frame extraction

```bash
python extract_frames.py
```

### Actor-independent 5-fold cross-validation

```bash
python train_5fold.py
```

### Grad-CAM visualization

```bash
python gradcam.py
```

## Code Availability

The source code for the proposed two-stage VGG16-based framework is publicly available in this repository.

The repository includes:

- frame extraction,
- actor-independent five-fold cross-validation,
- VGG16 fine-tuning,
- video-level aggregation using majority voting,
- Grad-CAM visualization.

The RAVDESS dataset is publicly available and can be obtained from its official repository.
