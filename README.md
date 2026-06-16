# Two-Stage VGG16 Emotion Recognition on RAVDESS

## Overview

This repository contains the source code associated with the following article:

**A Two-Stage Deep Learning Approach for Facial Emotion Recognition in RAVDESS Videos**

**Authors:** Achraf Jallaglag, My Abdelouahed Sabri, Ali Yahyaouy, and Abdellah Aarab

**Journal:** Scientific Reports (Springer Nature)

**DOI:** To be added upon publication.

The proposed framework investigates video-based facial emotion recognition using a lightweight two-stage architecture. Spatial features are extracted using a fine-tuned VGG16 network, while video-level predictions are obtained through confidence-based majority voting.

---

## Repository Structure

```text
ravdess_vgg16_code/
├── model.py
├── train_5fold.py
├── gradcam.py
├── utils.py
├── requirements.txt
└── README.md
```

---

## Method Summary

- **Input:** Facial frames extracted from RAVDESS videos.
- **Spatial Feature Extraction:** Fine-tuned VGG16 network.
- **Frame-Level Classification:** Softmax classifier.
- **Video-Level Aggregation:** Confidence-based majority voting.
- **Interpretability:** Grad-CAM visualization.
- **Evaluation Protocol:** Actor-independent 5-fold cross-validation.
- **Dataset:** Video-only speech subset of the RAVDESS dataset.

The proposed framework provides a lightweight and computationally efficient solution for video-based facial emotion recognition.

---

## Dataset

The experiments were conducted on the video-only speech subset of the RAVDESS dataset.

The dataset is not redistributed in this repository and can be obtained from its official source.

---

## Installation

Python ≥ 3.8

```bash
pip install -r requirements.txt
```

---

## Usage

### Actor-Independent 5-Fold Cross-Validation

```bash
python train_5fold.py
```

### Grad-CAM Visualization

```bash
python gradcam.py
```

---

## Reproducibility

This repository contains the scripts required to reproduce the training, evaluation, and Grad-CAM visualization procedures described in the paper.

---

## Code Availability

The source code for the proposed two-stage VGG16-based framework is publicly available in this repository.

The repository includes:

- actor-independent five-fold cross-validation,
- VGG16 fine-tuning,
- confidence-based majority voting,
- Grad-CAM visualization.

The RAVDESS dataset is publicly available and can be obtained from its official repository.

After publication, this repository will be archived on Zenodo and assigned a DOI.

---

## Citation

If you use this code, please cite:

```bibtex
@article{jallaglag2026emotion,
  title={A Two-Stage Deep Learning Approach for Facial Emotion Recognition in RAVDESS Videos},
  author={Jallaglag, Achraf and Sabri, My Abdelouahed and Yahyaouy, Ali and Aarab, Abdellah},
  journal={Scientific Reports},
  year={2026},
  doi={TO_BE_ADDED}
}
```

---

## Disclaimer

This code is provided for research and educational purposes only.

---

## Contact

**Achraf Jallaglag**  
Faculty of Sciences Dhar El Mahraz  
Sidi Mohamed Ben Abdellah University, Fez, Morocco

📧 **achraf.jallaglag@usmba.ac.ma**
