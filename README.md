Here is the complete, highly detailed `README.md` file. It perfectly integrates your final `train_hpc.py` script features (auto-installation, checkpoint resumption), the experimental logs from your CSV files, and the exact 7-Experiment narrative we developed.

You can copy and paste this directly into your GitHub repository.

---

# CIFAR-10 Efficient ResNet Architecture Challenge

**Course:** Fall 2026 Deep Learning - Project 1  
**Team Members:** Zirui Zheng, Hang Sun
**Final Kaggle Score:** **0.861** (Highest among our iterations)  
**Final Model Parameter Count:** **3,829,226** (Strictly under the 5M limit)

## 📖 1. Project Overview

This repository contains the complete codebase and training scripts for the Fall 2026 Kaggle CIFAR-10 Image Classification Challenge.

The goal was to design and train a modified Residual Network (ResNet) **from scratch** under a strict constraint of **≤ 5 million trainable parameters**. Through a rigorous 7-experiment iterative process, we discovered that standard ImageNet techniques (like Bottlenecks and overly aggressive augmentations) severely hurt performance on 32x32 images. Our final solution utilizes a **"Wide and Shallow" BasicBlock architecture** with a carefully scheduled **3-Phase training lifecycle (Warmup + CutMix + Label Smoothing)** over **200 epochs**, achieving our highest Kaggle score.

---

## 📂 2. Repository Structure & Data Setup

Due to file size limits, the dataset is not included in this repository. Please structure your environment as follows:

```text
├── train_hpc.py            # Main training script for final Exp 7 model (HPC-ready)
├── README.md               # Project documentation
├── data.zip                # (Optional) Zipped dataset; script will auto-extract
└── cifar_data/             # Dataset directory (automatically generated if zipped)
    ├── train_labels.csv    # Training labels
    ├── train_images/       # 50,000 training images
    ├── test.csv            # Test set IDs and filenames
    └── test_images/        # Held-out test images
```

---

## ⚙️ 3. Environment Setup (Auto-Installation)

To ensure maximum reproducibility without requiring TAs to manually install packages, `train_hpc.py` contains a built-in dependency checker.

You **do not** need to run `pip install` manually. The script automatically detects and installs missing libraries (`numpy`, `pandas`, `Pillow`, `scikit-learn`, `torch`, `torchvision`) at runtime.

**Note on Hardware:** The code automatically detects CUDA and uses **Automatic Mixed Precision (AMP)** for fast GPU training. If no GPU is found, it gracefully falls back to CPU.

---

## 🧠 4. Final Architecture & Hyperparameters (Exp 7)

We abandoned the `Bottleneck` structure (which loses spatial details on 32x32 images) and returned to a **Wide & Shallow** standard ResNet-18 variant.

- **Architecture:** BasicBlock ResNet
- **Channels:** `[48, 96, 192, 256]` (Expansion = 1)
- **Total Parameters:** `3,829,226` (Verified by script)
- **Optimizer:** SGD (`lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`)
- **Dropout:** `0.2` (Placed before the final Fully Connected layer)

### The 3-Phase Training Lifecycle (Crucial for Exp 7):

1.  **Phase 1: Warmup (Epoch 1 - 5):** LR linearly scales from `0.01` to `0.1` to stabilize early gradients.
2.  **Phase 2: Deep Training (Epoch 6 - 180):** Cosine Annealing + **CutMix (`alpha=0.5`)**. Forces the model to learn robust features from incomplete patches.
3.  **Phase 3: Cleanup (Epoch 181 - 200):** CutMix is **OFF**. **Label Smoothing (`0.1`)** is turned ON to prevent catastrophic overfitting on clean images and calibrate confidence.

---

## 🚀 5. How to Reproduce (HPC / Kaggle Guide)

1. Clone this repository.
2. Place your `cifar_data/` folder (or `data.zip`) in the root directory.
3. Execute the training script:
   ```bash
   python train_hpc.py
   ```
4. **Seamless Checkpoint Resumption (For HPC):**
   If your HPC job gets preempted or times out, **do not panic**. The script automatically saves a `latest.pth` checkpoint after **every single epoch**. If you restart the script, it will automatically detect this file and **resume exactly from the last completed epoch** (loading model weights, optimizer, and scheduler states) without losing any progress.
5. **Outputs:** All results are neatly organized in `./output/project1_runs/`:
   - `submission_Exp7_...csv`: Final Kaggle submission (with TTA applied).
   - `probs_Exp7_...npy`: Probability matrices saved for future Ensembling.
   - `epoch_log.csv`: Full training/validation metrics for every epoch.
   - `experiments.csv`: Final summary of parameters, time, and best accuracy.

---

## 🔬 6. Key Takeaways & Experiment Logs

To thoroughly understand the impact of different architectural and regularization choices, we conducted 7 distinct experiments.

### 📊 Experiment Summary Table (from `experiments.csv`)

| Exp       | Architecture & Key Strategies           | Params    | Avg Time/Epoch | Kaggle Score | Key Takeaway / Lesson Learned                                                                     |
| :-------- | :-------------------------------------- | :-------- | :------------- | :----------- | :------------------------------------------------------------------------------------------------ |
| **Exp 1** | Baseline (3-stage, ColorJitter, 30 Ep)  | 4.62M     | 24.0s          | **0.813**    | **Local scores are deceptive.** Severe overfitting (Train 98% vs Val 93%).                        |
| **Exp 2** | 4-Stage + CutMix + RandErasing (30 Ep)  | 4.90M     | 19.0s          | **0.806**    | Architecture changes alone don't fix generalization.                                              |
| **Exp 3** | Stratified Split + TTA (30 Ep)          | 4.90M     | 19.6s          | **0.814**    | **Epoch limit is the bottleneck.** Data pipeline was clean; 30 epochs was too short.              |
| **Exp 4** | ResNet-18 Variant (100 Ep, CutMix)      | 4.37M     | 17.6s          | **0.838**    | **Long training works.** Kaggle score jumped by 2.4%.                                             |
| **Exp 5** | Bottleneck + Extreme Aug (100 Ep)       | 4.63M     | 26.5s          | **0.622**    | **Over-regularization disaster.** Mixup + RandAug(mag=9) destroyed spatial semantics.             |
| **Exp 6** | Bottleneck + Mild Aug (100 Ep)          | 4.63M     | 24.2s          | **0.830**    | **Goldilocks zone.** Proper regularization budget (CutMix + soft RandAug) restored real learning. |
| **Exp 7** | **BasicBlock (Wide) + 200 Ep + Warmup** | **3.83M** | **15.4s**      | **0.861**    | **Champion Model.** "Wide and shallow" > "Deep and narrow" for 32x32 images.                      |

### 📈 Critical Training Logs & Diagnostics (from `epoch_log.csv`)

Here are the specific log snapshots that proved our hypotheses during the training lifecycle:

#### 🔍 1. The "Fake High Score" (Exp 1)

- **Log:** `Epoch 30 | Train Loss: 0.0539, Acc: 98.54% | Val Loss: 0.2047, Acc: 93.56%`
- **Diagnostic:** Huge gap between Train (98.5%) and Val (93.5%). The model memorized the pixels. **Kaggle dropped to 0.81.**

#### 🚀 2. The "Epoch Limit" Breakthrough (Exp 4)

- **Log:** `Epoch 80 | Train Acc: 71.76%` \(\rightarrow\) `Epoch 81 | Train Acc: 93.29%`
- **Diagnostic:** Turning off CutMix at Epoch 81 caused the Train Acc to skyrocket. Proved that 100 epochs was necessary and CutMix was effectively forcing feature learning.

#### ⚠️ 3. The "Over-Regularization" Catastrophe (Exp 5)

- **Log:** `Epoch 100 | Train Loss: 1.6952, Acc: 50.34% | Val Loss: 1.0773, Acc: 75.92%`
- **Diagnostic:** Massive _negative Gap_ (Val Loss < Train Loss). The model was "underfitting" because Mixup(alpha=1.0) + RandAugment(mag=9) turned images into unrecognizable hybrids. **Kaggle crashed to 0.622.**

#### ✨ 4. The "Textbook Convergence" & Gap Flip (Exp 6)

- **Log:** `Epoch 90 | Train Loss: 0.9865, Acc: 70.51% | Val Loss: 0.3102, Acc: 91.38%` (CutMix ON)
- **Log:** `Epoch 91 | Train Loss: 0.1976, Acc: 93.64% | Val Loss: 0.2539, Acc: 91.42%` (CutMix OFF)
- **Diagnostic:** The moment CutMix turned off, the model saw clean images. The negative Gap instantly flipped to a healthy positive (+0.05). The model proved it wasn't memorizing; it was just "blindfolded" by CutMix during training.

#### 🏆 5. The Final Champion: 200 Epochs & 3-Phase Transition (Exp 7)

- **Phase 1 (Warmup):** `Epoch 5 | Val Acc: 70.50%`
- **Phase 2 (CutMix Deep Training):** `Epoch 180 | Train Loss: 0.6812 | Val Acc: 94.98%`
- **Phase 3 (CutMix OFF + Label Smoothing):** `Epoch 181 | Train Loss: 0.5276 (dropped from 0.68) | Val Acc: 95.18%`
- **Final Epoch:** `Epoch 200 | Train Loss: 0.5094 | Val Acc: 95.46%`
- **Diagnostic:** By scaling to 200 epochs with a flattened Cosine decay, the model reached a much smoother local minimum. Switching off CutMix at Epoch 181 for the final 20 epochs, while introducing Label Smoothing, allowed the model to calibrate its confidence on clean images, pushing the **Kaggle score to a peak of 0.861**.

---

## 📚 7. References

1. Cubuk, E. D., Zoph, B., Shlens, J., & Le, Q. V. (2020). RandAugment: Practical Automated Data Augmentation with a Reduced Search Space. IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). [arXiv:1909.13719](https://arxiv.org/abs/1909.13719) (Refer 1: Used in Exp 5 & 6 to evaluate aggressive vs. mild augmentation).

2. Pham, H., & Le, Q. V. (2021). AutoDropout: Learning Dropout Patterns to Regularize Deep Networks. arXiv preprint. [arXiv:2101.01761](https://arxiv.org/abs/2101.01761) (Refer 2: Inspired our discussion on DropPath vs. standard Dropout in Exp 5 & 6).
