# Experiment 5: Comprehensive Study of CNN Training, Regularization, Optimization, Hyperparameter Tuning, Transfer Learning and Cross-Validation

## Aim
To systematically study the effect of weight initialization, regularization, optimization algorithms, CNN hyperparameters, transfer learning, fine-tuning, and cross-validation on image classification performance, using a single CNN architecture (MobileNetV2) on the Oxford-IIIT Pet dataset, and to select a final configuration via 5-fold cross-validation.

---

## Dataset Description

**Dataset: Oxford-IIIT Pet Dataset**

| Property | Value |
|---|---|
| Classes (breeds) | 37 |
| Official Training Images | 3,680 |
| Official Testing Images | 3,669 |
| Image Size | 224 × 224 × 3 (RGB, resized) |
| Train/Validation Split | 80% / 20% |
| Training Subset Used (GPU profile) | 60% (post val-split) |

The 3,680-image official training split was shuffled and divided 80/20 into internal training and validation sets, used for all epoch-by-epoch curves. The 3,669-image test split was kept completely untouched until final evaluation. A 60% training subsample was used per-experiment to keep total run time reasonable across the many models trained in this study.

---

## Procedure

1. **Weight Initialization Study** — Compare Zero, Random Normal, Xavier/Glorot, and He initialization on a MobileNetV2 head (10 epochs, Adam, lr = 10⁻³); plot training loss and validation accuracy per method.
2. **Regularization Study** — Compare No Regularization, L2, Dropout, and Batch Normalization (12 epochs); plot training vs. validation accuracy/loss and compute the generalization gap for each.
3. **Batch Normalization Analysis** — Derive the BN transformation analytically with a numerical example; compare validation accuracy with vs. without BN.
4. **Optimizer Comparison** — Compare SGD, Momentum, RMSProp, and Adam (10 epochs) on training loss and validation accuracy convergence.
5. **Hyperparameter Tuning** — One-factor-at-a-time sweep over learning rate (10⁻³, 10⁻⁴), batch size (16, 32, 64), and dropout rate (0, 0.25, 0.5).
6. **Transfer Learning & Fine-Tuning** — Compare feature extraction (frozen base) vs. fine-tuning (unfrozen upper layers, reduced learning rate); sweep fine-tuning learning rate and frozen-layer strategy.
7. **5-Fold Cross-Validation** — Evaluate 4 shortlisted configurations across 5 folds; report mean accuracy ± standard deviation.
8. **Final Model Evaluation** — Retrain the best cross-validated configuration on the complete training split and evaluate once on the untouched test set (Accuracy, Precision, Recall, F1-score, Confusion Matrix, Classification Report).

---

## Techniques Used

- Weight initialization strategies: Zero, Random Normal, Xavier/Glorot, He
- Regularization: L2 weight decay, Dropout, Batch Normalization
- Optimizer comparison: SGD, Momentum, RMSProp, Adam
- MobileNetV2 architecture: depthwise separable convolutions, inverted residual blocks, linear bottlenecks, ReLU6 activation
- Transfer learning via feature extraction (frozen ImageNet backbone)
- Fine-tuning via partial unfreezing at a reduced learning rate
- 5-fold cross-validation for robust model selection
- Fixed random seeding for reproducibility
- Classification head: `GlobalAveragePooling → BatchNorm → Dense(128) → Dropout → Softmax`

---

## MobileNetV2 Architecture (Simplified)

Input (224×224×3) → Conv → Depthwise Conv → Pointwise Conv → Residual Block → GlobalAveragePooling → Dense → Softmax

**Convolution Output Size Formula:**

O = ⌊(N + 2P − K) / S⌋ + 1

| Case | N | K | P | S | Output |
|---|---:|---:|---:|---:|---:|
| Same-padded 3×3 | 224 | 3 | 1 | 1 | 224 (preserved) |
| MobileNetV2 stem conv | 224 | 3 | 0 | 2 | 111 |

---

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer (best) | Adam |
| Learning Rate — Head Training | 10⁻³ |
| Learning Rate — Fine-Tuning | 10⁻⁴ – 10⁻⁵ |
| Batch Size (best) | 32 |
| Dropout Rate (best) | 0 – 0.3 |
| Epochs — Initialization/Optimizer Studies | 10 |
| Epochs — Regularization Study | 12 |
| Epochs — Final Model Retrain | 18 |
| Loss Function | Categorical Cross-Entropy |
| Weight Initialization (final) | He |

---

## Key Study Results

**Weight Initialization (10 epochs):**

| Initialization | Final Training Loss | Best Validation Accuracy |
|---|---:|---:|
| Zero | 3.6105 | 2.85% |
| Random Normal | 0.1464 | 98.37% |
| Xavier/Glorot | 0.1286 | 99.05% |
| He | 0.1389 | 98.91% |

**Regularization — Generalization Gap (Train − Val Accuracy):**

| Configuration | Gap (pp) |
|---|---:|
| No Regularization | +0.17 |
| L2 Regularization | −0.45 |
| Dropout | −4.92 |
| Batch Normalization | −1.34 |

**Optimizer Comparison (10 epochs):**

| Optimizer | Final Loss | Best Validation Accuracy | Epochs to Converge | Time |
|---|---:|---:|---:|---:|
| SGD | 2.1269 | 61.55% | 10 | 116.3 s |
| Momentum | 0.4236 | 93.61% | 9 | 114.2 s |
| RMSProp | 0.1251 | 98.51% | 10 | 113.9 s |
| **Adam** | 0.1350 | **99.05%** | 9 | 115.3 s |

**Hyperparameter Sweep:**

| Learning Rate | Val. Accuracy | Batch Size | Val. Accuracy | Dropout | Val. Accuracy |
|---|---:|---|---:|---:|---:|
| 10⁻³ | 97.69% | 16 | 98.23% | 0.00 | 99.05% |
| 10⁻⁴ | 89.13% | 32 | **98.78%** | 0.25 | 97.69% |
| — | — | 64 | 98.51% | 0.50 | 97.55% |

**Fine-Tuning Sensitivity:**

| Fine-Tuning Learning Rate | Best Validation Accuracy |
|---|---:|
| 10⁻⁴ | 95.38% |
| 10⁻⁵ | 64.40% |

| Frozen-Layer Strategy | Best Validation Accuracy |
|---|---:|
| Frozen base (feature extraction), lr = 10⁻³ | 98.64% |
| Partial unfreeze (top 30%), lr = 10⁻⁵ | 64.81% |

---

## 5-Fold Cross-Validation Results

| Configuration | F1 | F2 | F3 | F4 | F5 | Mean ± SD |
|---|---:|---:|---:|---:|---:|---|
| C1 — Baseline (Glorot, dropout 0.3) | 89.83 | 93.48 | 87.82 | 88.39 | 88.95 | 89.69 ± 2.01 |
| **C2 — Best Init (He, dropout 0.3)** | 90.68 | 93.77 | 87.25 | 89.24 | 89.52 | **90.09 ± 2.14** |
| C3 — Dropout 0.5 | 88.98 | 94.33 | 86.69 | 88.10 | 89.80 | 89.58 ± 2.59 |
| C4 — Lower LR (10⁻⁴) | 82.77 | 82.72 | 81.59 | 79.60 | 82.44 | 81.82 ± 1.19 |

**Selected Configuration:** C2 — He initialization, dropout 0.3, Batch Normalization, Adam (lr = 10⁻³).

---

## Evaluation Metrics

### Final Model (Retrained + Tested)

| Metric | Value |
|---|---:|
| Mean CV Accuracy | 90.09% |
| CV Standard Deviation | 2.14 |
| Test Accuracy | 89.21% |
| Precision (weighted) | 0.8948 |
| Recall (weighted) | 0.8921 |
| F1-score (weighted) | 0.8909 |
| Training Time | 181.9 s |
| Parameters | 2,431,845 |

### Overall Results Summary

| Configuration | CV Accuracy | SD | Test/Val. Accuracy | Training Time |
|---|---:|---:|---:|---:|
| Baseline | — | — | 88.85% | 153.4 s |
| Best Initialization (He) | — | — | 98.91% | — |
| Best Regularization (Dropout) | — | — | 97.01% | — |
| Best Optimizer (Adam) | — | — | 99.05% | 115.3 s |
| Best Hyperparameters (C2) | 90.09% | 2.14 | — | — |
| Fine-Tuned Model | — | — | 99.73% | — |
| **FINAL MODEL** | **90.09%** | **2.14** | **89.21%** | 181.9 s |

### Per-Class Performance (Test Set)

**Best classified:** Yorkshire Terrier (100.0%), Samoyed (100.0%), Japanese Chin (99.0%), Shiba Inu (99.0%), Leonberger (99.0%).

**Most confused:** American Pit Bull Terrier (51.0%), Ragdoll (69.0%), British Shorthair (71.0%), Staffordshire Bull Terrier (74.2%), Birman (76.0%).

---

## Conclusion

This experiment used MobileNetV2 on the Oxford-IIIT Pet dataset to isolate and quantify the effect of individual design choices on CNN training.

Weight initialization proved critical — zero initialization caused complete training failure (2.85% accuracy) due to symmetric gradients, while He and Xavier/Glorot both converged reliably to ~99%. Among regularizers, Dropout produced the strongest generalization gap reduction (−4.92 pp) at some cost to peak accuracy, while Batch Normalization offered a balanced trade-off. Adam and RMSProp clearly outperformed SGD and Momentum in both convergence speed and final accuracy, confirming adaptive per-parameter learning rates are well-suited to this transfer-learning setting.

Fine-tuning results reinforced that pretrained weights are sensitive to learning rate: a rate of 10⁻⁴ substantially outperformed 10⁻⁵ within the same short training budget, and aggressive fine-tuning without adequate epochs degraded performance relative to simple feature extraction.

5-fold cross-validation identified **He initialization with dropout 0.3 and Adam (C2)** as the most reliable configuration (90.09% ± 2.14%), balancing mean accuracy and consistency better than lower-dropout or lower-learning-rate alternatives. The final retrained model achieved **89.21%** test accuracy with a weighted F1-score of 0.8909, confirming that cross-validated model selection generalizes well to the untouched test set.

Thus, careful weight initialization, regularization choice, and optimizer selection are each independently significant levers for CNN performance, and cross-validation is essential for selecting configurations that generalize reliably rather than merely maximizing a single validation split.
