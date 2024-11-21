# RANZCR CLiP - Catheter and Line Position Challenge

## Kaggle Competition
**Dates:** February 2021 - March 2021  
**Competition Page:** [RANZCR CLiP - Catheter and Line Position Challenge](https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-position-challenge)

---

## Project Overview
This project focuses on leveraging deep learning to classify catheter and line positions in medical X-ray images for the **RANZCR CLiP Challenge**. The proposed solution utilizes **Seresnet152d** and implements a **perturbation-based knowledge distillation technique** to achieve high classification performance.

### Key Achievements:
- **Model:** Seresnet152d, enhanced with perturbation-based knowledge distillation.
- **Framework:** PyTorch.
- **Metric:** Achieved an **AUC score of 95.1**.

---

## Approach and Methodology

### Dataset
- **Source:** Kaggle competition dataset.
- **Type:** Medical X-ray images with labeled annotations for catheter and line positions.
- **Preprocessing Steps:**
  - Histogram equalization for contrast enhancement.
  - Data augmentations: random cropping, random rotations, grayscale conversion, and normalization.

### Model Architecture
- **Backbone:** Seresnet152d (pretrained via the `timm` library).
- **Enhancements:** Knowledge transfer from a Teacher model to a Student model using a perturbation-based method.

### Training and Evaluation
- **Loss Function:** Binary Cross Entropy with Logits (BCEWithLogitsLoss).
- **Optimizer:** Adam with learning rate scheduling.
- **Training Phases:**
  - **Phase 2:** Teacher model pretraining.
  - **Phase 3:** Fine-tuning and knowledge distillation to the Student model.
- **Evaluation Metric:** AUC (Area Under the Curve).

---

## Project Files

| File Name                                | Description                                      |
|------------------------------------------|--------------------------------------------------|
| `phase_2.py`                             | Teacher model training script.                  |
| `phase_3.py`                             | Student model training and distillation script. |
| `final.py`                               | Final training script for evaluation.           |
| `teacher_11_label_complete.csv`          | Annotations for the Teacher model.              |
| `ultimate_11_label_teacher_deleted_concatenated.csv` | Annotations for the Student model.              |
| `teacher_30epochs_624_SDG`               | Pretrained Teacher model weights.               |
| `README.md`                              | Project documentation (this file).              |

---

## Results
The final model achieved an **AUC score of 95.1** using the perturbation-based knowledge distillation technique.

---

## Running the Code

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/RANZCR-CLiP.git
   cd RANZCR-CLiP
