# Two-Stage VGG16 Emotion Recognition on RAVDESS

This repository contains a compact implementation of the code used for the paper:

**A Two-Stage Deep Learning Approach for Facial Emotion Recognition in RAVDESS Videos**

The repository includes only the essential scripts required for reproducibility:

1. Frame extraction from RAVDESS videos.
2. Actor-independent 5-fold cross-validation.
3. Fine-tuned VGG16 frame-level classification.
4. Video-level aggregation using confidence-based majority voting.
5. Grad-CAM visualization.

## Repository structure

```text
ravdess_vgg16_code/
├── config.py
├── utils.py
├── extract_frames.py
├── model.py
├── train_5fold.py
├── gradcam.py
├── requirements.txt
└── README.md
```

## Dataset

The dataset used is the video-only speech subset of RAVDESS. The dataset is not redistributed in this repository. Download it from the official source and update `DATA_DIR` in `config.py`.

Expected input: recursive folders containing RAVDESS video files such as:

```text
01-01-03-01-01-01-01.mp4
01-01-04-01-01-01-01.mp4
...
```

## Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Edit paths

Open `config.py` and set:

```python
DATA_DIR = Path("/path/to/RAVDESS/video_speech")
```

### 3. Extract frames

```bash
python extract_frames.py
```

### 4. Run actor-independent 5-fold cross-validation

```bash
python train_5fold.py
```

The script saves:

```text
outputs/models/fold_X_best_vgg16_ravdess.keras
outputs/figures/actor_independent_5fold_results.csv
```

### 5. Generate Grad-CAM

```bash
python gradcam.py \
  --model /path/to/best_model.keras \
  --image /path/to/frame.jpg \
  --true_label fearful \
  --output gradcam_fearful.png
```

## Code Availability statement

The source code for the proposed two-stage VGG16-based RAVDESS facial emotion recognition pipeline is available in this repository. The code includes frame extraction, actor-independent five-fold validation, VGG16 fine-tuning, video-level aggregation, and Grad-CAM visualization.

After uploading this repository to GitHub, archive it on Zenodo and add the Zenodo DOI to the manuscript.
