# Experiment 4: Comparative Study of Deep Convolutional Neural Network Architectures Using Transfer Learning

## Aim
To study the evolution of deep CNN architectures (LeNet-5, AlexNet, VGG16, GoogleNet, ResNet), implement transfer learning and fine-tuning using pretrained ImageNet backbones (VGG16, ResNet50) on the CIFAR-10 dataset, and compare their classification performance against CNNs trained from scratch.

---

## Dataset Description

**Dataset: CIFAR-10**

| Property | Value |
|---|---|
| Training Images | 50,000 |
| Testing Images | 10,000 |
| Classes | 10 |
| Image Size | 32 × 32 × 3 (RGB) |
| Validation Split | 90:10 (fixed seed = 42) |
| Pixel Value Range | Normalized to [0.000, 1.000] |

**Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

**Actual Split:**

| Split | Images | Shape |
|---|---:|---|
| Training | 45,000 | (45000, 32, 32, 3) |
| Validation | 5,000 | (5000, 32, 32, 3) |
| Testing | 10,000 | (10000, 32, 32, 3) |

---

## Procedure

1. **Dataset Preparation** — Load CIFAR-10 via TensorFlow/Keras, normalize pixel values to [0,1], display ten sample images, print dataset dimensions.
2. **Transfer Learning Setup** — Load VGG16 and ResNet50 with pretrained ImageNet weights (`include_top=False`), remove the original classifier, freeze the convolutional base, resize inputs from 32×32 to 96×96, apply backbone-specific `preprocess_input`.
3. **Classification Head** — Attach `GlobalAveragePooling2D → Dense(ReLU) → Dropout(0.3) → Dense(10, Softmax)` to each frozen backbone.
4. **Model Training** — Train the frozen-stage models with Adam optimizer, learning rate 0.001, batch size 64, for 8 epochs (Categorical/Sparse Cross-Entropy loss).
5. **Fine-Tuning** — Unfreeze the last convolutional block (VGG16: last 6 layers; ResNet50: last 10 layers) and retrain for 6 further epochs at a reduced learning rate (1×10⁻⁵); compare accuracy before and after fine-tuning.
6. **Baseline Models** — Train LeNet-5, AlexNet-Mini, and ResNet-Mini from scratch on native 32×32 CIFAR-10 images for comparison.
7. **Model Evaluation** — Compute Accuracy, Precision, Recall, F1-score, Classification Report, and Confusion Matrix for all models on the held-out 10,000-image test set.
8. **Hyperparameter Study** — Compare Adam vs. SGD optimizers; grid search over learning rate, dense units, and frozen-layer strategy (all frozen vs. partial unfreeze).

---

## Techniques Used

- Transfer learning with frozen convolutional bases (VGG16, ResNet50)
- Fine-tuning via selective unfreezing of upper convolutional layers
- GlobalAveragePooling2D + Dense(ReLU) + Dropout(0.3) classification head
- Backbone-specific input preprocessing (`preprocess_input`)
- Mixed-precision training (`mixed_float16`) on dual Tesla T4 GPUs
- Optimizer comparison (Adam vs. SGD with momentum)
- Hyperparameter grid search (learning rate, dense units, frozen-layer strategy)
- Categorical Cross-Entropy / Sparse Categorical Cross-Entropy loss with Adam optimizer
- Fixed random seed (42) for NumPy, Python, and TensorFlow reproducibility

---

## Transfer Learning Architecture

**Head (attached to both VGG16 and ResNet50):** Frozen Backbone → GlobalAveragePooling2D → Dense(ReLU) → Dropout(0.3) → Dense(10, Softmax)

**VGG16 Transfer-Learning Model Summary:**

| Property | Value |
|---|---:|
| Total Parameters | 14,848,586 (56.64 MB) |
| Trainable (frozen stage) | 133,898 (523.04 KB) |
| Non-trainable (frozen stage) | 14,714,688 (56.13 MB) |

---

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate — Frozen Stage | 0.001 |
| Learning Rate — Fine-Tuning Stage | 1 × 10⁻⁵ |
| Batch Size | 64 |
| Epochs — Baseline Models | 12 |
| Epochs — Frozen Stage | 8 |
| Epochs — Fine-Tuning Stage | 6 |
| Epochs — Optimizer Comparison | 5 |
| Epochs — Hyperparameter Search | 4 |
| Loss Function | Sparse Categorical Cross-Entropy |
| Transfer-Learning Image Size | 96 × 96 |
| Baseline Image Size | 32 × 32 |

---

## Fine-Tuning Results

| Model | Accuracy Before Fine-Tuning | Accuracy After Fine-Tuning | Improvement |
|---|---:|---:|---:|
| VGG16 | 83.10% | **90.26%** | +7.16 pp |
| ResNet50 | 84.41% | **86.14%** | +1.73 pp |

> Fine-tuning improved both backbones, with a much larger gain for VGG16, suggesting its ImageNet features benefited more from adapting to the low-resolution CIFAR-10 domain, while ResNet50's residual features were already closer to optimal in the frozen state.

---

## Evaluation Metrics

### Best Model: VGG16 (Fine-Tuned)

| Metric | Value |
|---|---:|
| Training Accuracy | 96.39% |
| Testing Accuracy | 90.26% |
| Precision | 0.9035 |
| Recall | 0.9026 |
| F1-score | 0.9028 |
| Training Time | 290.3 s (fine-tuning); 141.1 s (frozen) |
| Total Parameters | 14,848,586 |

---

## Comparison of CNN Architectures

| Model | Parameters | Test Accuracy | Precision | Recall | F1-score | Training Time |
|---|---:|---:|---:|---:|---:|---:|
| LeNet-5 | 83,126 | 54.26% | 0.5481 | 0.5426 | 0.5403 | 232.8 s |
| AlexNetMini | 6,976,842 | 75.26% | 0.7575 | 0.7526 | 0.7486 | 306.1 s |
| ResNetMini | 681,226 | 47.26% | 0.5552 | 0.4726 | 0.4317 | 141.1 s |
| VGG16 (Frozen, TL) | 14,848,586 | 83.10% | 0.8323 | 0.8310 | 0.8306 | 141.1 s |
| **VGG16 (Fine-Tuned, TL)** | 14,848,586 | **90.26%** | **0.9035** | **0.9026** | **0.9028** | 290.3 s |
| ResNet50 (Frozen, TL) | — | 84.41% | 0.8455 | 0.8441 | 0.8440 | 91.1 s |
| ResNet50 (Fine-Tuned, TL) | — | 86.14% | 0.8614 | 0.8614 | 0.8613 | 194.3 s |

---

## Hyperparameter Study (VGG16)

**Optimizer Comparison (5 epochs, 96×96 input):**

| Optimizer | Test Accuracy | F1-score | Training Time |
|---|---:|---:|---:|
| Adam (lr = 0.001) | 82.79% | 0.8278 | 89.0 s |
| SGD (lr = 0.01, momentum = 0.9) | 80.79% | 0.8079 | 88.1 s |

**Hyperparameter Search (4 epochs each):**

| Learning Rate | Dense Units | Frozen Layers | Validation Accuracy | Time |
|---|---:|---|---:|---:|
| 0.0010 | 128 | All | 0.8330 | 73.4 s |
| 0.0010 | 256 | All | 0.8402 | 72.8 s |
| 0.0001 | 256 | Partial (last 4 unfrozen) | **0.9098** | 89.0 s |

> Partially unfreezing the convolutional base combined with a lower learning rate gave the largest gain (+6.96 points over the best fully-frozen configuration), confirming backbone adaptation is more valuable than widening the classifier head.

---

## Conclusion

CIFAR-10 was used to compare classic and modern CNN architectures trained from scratch against transfer-learned VGG16 and ResNet50 backbones.

The from-scratch baselines showed a clear capacity hierarchy: LeNet-5 plateaued at 54.26% due to its shallow depth, AlexNet-Mini reached 75.26% with good generalization, and ResNet-Mini underperformed (47.26%) as its residual design was not trained stably in the limited epochs available.

Transfer learning substantially outperformed all baselines. With the convolutional base frozen, VGG16 and ResNet50 achieved 83.10% and 84.41% test accuracy respectively, using only a lightweight classification head. Fine-tuning the last convolutional block further improved both models — VGG16 reached **90.26%** (F1-score 0.9028), the best result overall, while ResNet50 improved more modestly to 86.14%, indicating its frozen features were already closer to optimal for this task.

Thus, transfer learning with selective fine-tuning is significantly more effective and parameter-efficient than training deep CNNs from scratch on small, low-resolution datasets like CIFAR-10, with the choice of backbone and the extent of unfreezing being the primary levers for final accuracy.
