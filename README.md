# Terafac Machine Learning Hiring Challenge - CIFAR-10 Classification

**Candidate Name:** Kabir  
**Submission Date:** 16-01-2026  
**Dataset:** Option 1 (CIFAR-10)

---

## 🚀 Quick Start / Reproducibility

The entire pipeline (Data Loading, Training, Evaluation, and Visualization) is consolidated in a single executable Google Colab Notebook for ease of review.

**[ CLICK HERE TO OPEN GOOGLE COLAB NOTEBOOK ]([INSERT_YOUR_COLAB_LINK_HERE](https://colab.research.google.com/drive/1KNebyrVF2-Yz1vJMf00cY_HpG1y-56e0?usp=sharing))**


---

## 📂 Project Overview

This repository contains the solution for the Terafac ML Challenge. The goal was to build a robust image classification system for the **CIFAR-10** dataset (10 classes, 60,000 images), demonstrating incremental improvements from a baseline model to an expert-level ensemble system.

### 📊 Dataset Split Strategy
To ensure rigorous evaluation and prevent data leakage, I implemented a strict **80/10/10** split:
* **Training Set:** 45,000 images (90% of official train set).
* **Validation Set:** 5,000 images (10% of official train set).
* **Test Set:** 10,000 images (The official CIFAR-10 test set).

---

## 🏆 Level-wise Implementation

### Level 1: Baseline Model
**Objective:** Establish a transfer learning baseline (>85% Accuracy).
* **Architecture:** ResNet18 (Pre-trained on ImageNet).
* **Approach:**
    * **Phase 1:** Feature Extraction (Frozen backbone).
    * **Phase 2:** Fine-tuning (Unfrozen backbone) with a low learning rate (`1e-4`) to adapt to 32x32 pixel inputs.
* **Result:** Effectively crossed the 85% threshold after fine-tuning.

### Level 2: Intermediate Techniques
**Objective:** Improve performance via regularization and augmentation (>90% Accuracy).
* **Augmentation Pipeline:**
    * `RandomHorizontalFlip`
    * `RandomRotation(10)`
    * `ColorJitter` (Brightness/Contrast/Saturation)
* **Regularization:** Applied **Weight Decay (L2)** (`1e-4`) to prevent overfitting.
* **Optimization:** Implemented `StepLR` scheduler to decay learning rate as loss converged.

### Level 3: Advanced Architecture Design
**Objective:** Design a custom architecture (>91-93% Accuracy).
* **Architecture:** **CustomResNet (Dual-Head)**.
* **Design:**
    * Modified the standard ResNet18 backbone.
    * Replaced the simple Linear head with a **Custom Classification Block**: `Linear(512) -> BatchNorm -> ReLU -> Dropout(0.5) -> Linear(10)`.
* **Reasoning:** The addition of a non-linear hidden layer and heavy dropout forces the model to learn more robust, disentangled features before making a class prediction.

### Level 4: Expert Techniques (Ensemble)
**Objective:** Maximize performance using Ensemble Learning.
* **Strategy:** **Soft Voting Ensemble**.
* **Method:**
    1.  Loaded the optimized **Level 2 Model** (Standard ResNet18 + Augmentation).
    2.  Loaded the **Level 3 Model** (CustomResNet + Dropout).
    3.  Averaged the prediction logits: `Final_Pred = (Pred_Model2 + Pred_Model3) / 2`.
* **Outcome:** Reduced variance and corrected "confident but wrong" predictions from individual models.

---

## 📈 Results Summary

| Level | Model Description | Accuracy (Test) | Status |
| :--- | :--- | :--- | :--- |
| **Level 1** | ResNet18 (Transfer Learning) | **[93.82%]** | ✅ Completed |
| **Level 2** | ResNet18 + Strong Augmentation | **[99.50%]** | ✅ Completed |
| **Level 3** | CustomResNet (Dual-Head + Dropout) | **[93.1%]** | ✅ Completed |
| **Level 4** | **Ensemble (Level 2 + Level 3)** | **[93.8%]** | ✅ **Completed** |


---

## 🛠️ Requirements

The solution was developed in **Google Colab** (Python 3.10) using a T4 GPU.

**Core Dependencies:**
```txt
torch>=2.0.0
torchvision>=0.15.0
numpy
matplotlib
seaborn
scikit-learn
