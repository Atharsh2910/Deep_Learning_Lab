# Experiment 3: Convolutional Neural Network (CNN) for Multi-Class Image Classification

## Aim
To implement and systematically optimize a Convolutional Neural Network (CNN) using TensorFlow/Keras on the CIFAR-10 dataset, exploring convolution operations, feature map visualization, pooling strategies, regularization techniques (Batch Normalization, Dropout, L2), data augmentation, and adaptive learning rate scheduling to achieve improved generalization over a baseline CNN.

---

## Dataset Description

**Dataset: CIFAR-10**

| Property | Value |
|---|---|
| Training Images | 50,000 |
| Testing Images | 10,000 |
| Classes | 10 |
| Image Size | 32 × 32 × 3 (RGB) |
| Pixel Value Range | [0, 255] → normalized to [0, 1] |

**Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

---

## Procedure

1. **Dataset Exploration** — Load CIFAR-10, print dataset dimensions, display ten sample images, plot class distribution across all 10 categories.
2. **Data Preprocessing** — Normalize pixel values to [0, 1], convert labels to one-hot vectors, verify tensor shapes.
3. **Analytical Exercises** — Compute convolution output size analytically and verify with Keras (`ZeroPadding2D + Conv2D`); calculate trainable parameters for a Conv2D layer with 64 filters on RGB input.
4. **Activation Function Comparison** — Train a simple 2-Conv model with ReLU vs Sigmoid for 5 epochs on CIFAR-10; compare test accuracy and training time.
5. **Feature Map Visualization** — Visualize at least 8 feature maps produced by the first convolutional layer for a sample image.
6. **Pooling Comparison** — Compare Max Pooling vs Average Pooling on a 2-Conv model for 3 epochs; observe the effect on output shape and accuracy.
7. **Baseline CNN** — Build a simple 2-block CNN: Conv(32) → Pool → Conv(64) → Pool → Flatten → Dense(128) → Softmax; train for 20 epochs with batch size 32, Adam optimizer.
8. **Optimized CNN** — Design a deeper 3-block CNN with progressive filter expansion (32 → 64 → 128), Batch Normalization, Dropout (0.25 conv, 0.50 dense), L2 regularization, and GlobalAveragePooling2D; apply on-the-fly data augmentation; train with adaptive callbacks (ReduceLROnPlateau, EarlyStopping, ModelCheckpoint).
9. **Model Evaluation** — Compute Accuracy, Precision, Recall, F1-score, per-class Classification Report, and Confusion Matrix for both models.

---

## Techniques Used

- Convolutional layers with Batch Normalization (Conv → BN → ReLU pattern)
- Max Pooling and GlobalAveragePooling2D
- Dropout regularization (0.25 after conv blocks, 0.50 after dense layer)
- L2 weight regularization (λ = 1e-4) on all Conv and Dense layers
- On-the-fly data augmentation via `ImageDataGenerator`
- Adaptive learning rate scheduling with `ReduceLROnPlateau`
- Early stopping with best-weight restoration (`EarlyStopping`)
- Best-model checkpointing (`ModelCheckpoint`)
- ReLU vs Sigmoid activation comparison
- Feature map visualization from intermediate layers
- Categorical Cross-Entropy loss with Adam optimizer

---

## Data Augmentation Configuration

| Augmentation | Value |
|---|---|
| Rotation Range | 15° |
| Width Shift | 0.1 |
| Height Shift | 0.1 |
| Zoom Range | 0.1 |
| Horizontal Flip | True |
| Fill Mode | Nearest |

---

## Optimized Model Architecture

**Architecture:** 3 × [Conv(F) → BN → ReLU → Conv(F) → BN → ReLU → MaxPool → Dropout(0.25)] → GlobalAveragePooling2D → Dense(256) → BN → Dropout(0.50) → Softmax

| Layer | Output Shape | Parameters |
|---|---|---:|
| Input | (None, 32, 32, 3) | 0 |
| Conv2D (32 filters) | (None, 32, 32, 32) | 896 |
| BatchNormalization | (None, 32, 32, 32) | 128 |
| Conv2D (32 filters) | (None, 32, 32, 32) | 9,248 |
| BatchNormalization | (None, 32, 32, 32) | 128 |
| MaxPooling2D + Dropout(0.25) | (None, 16, 16, 32) | 0 |
| Conv2D (64 filters) | (None, 16, 16, 64) | 18,496 |
| BatchNormalization | (None, 16, 16, 64) | 256 |
| Conv2D (64 filters) | (None, 16, 16, 64) | 36,928 |
| BatchNormalization | (None, 16, 16, 64) | 256 |
| MaxPooling2D + Dropout(0.25) | (None, 8, 8, 64) | 0 |
| Conv2D (128 filters) | (None, 8, 8, 128) | 73,856 |
| BatchNormalization | (None, 8, 8, 128) | 512 |
| Conv2D (128 filters) | (None, 8, 8, 128) | 147,584 |
| BatchNormalization | (None, 8, 8, 128) | 512 |
| MaxPooling2D + Dropout(0.25) | (None, 4, 4, 128) | 0 |
| GlobalAveragePooling2D | (None, 128) | 0 |
| Dense (256, ReLU) | (None, 256) | 33,024 |
| BatchNormalization | (None, 256) | 1,024 |
| Dropout (0.50) | (None, 256) | 0 |
| Dense (10, Softmax) | (None, 10) | 2,570 |

**Total**: 325,418

**Trainable**: 324,010

**Non-trainable**: 1,408

---

## Training Configuration

| Parameter | Baseline CNN | Optimized CNN |
|---|---|---|
| Optimizer | Adam | Adam |
| Initial Learning Rate | Default (0.001) | 0.001 |
| Batch Size | 32 | 64 |
| Epochs | 20 (fixed) | Up to 60 (EarlyStopping) |
| LR Reduce Factor | — | 0.5 |
| LR Patience | — | 4 epochs |
| Early Stop Patience | — | 10 epochs |
| Data Augmentation | None | Rotation, flip, shift, zoom |
| Loss Function | Categorical Cross-Entropy | Categorical Cross-Entropy |

---

## Activation Function Comparison (ReLU vs Sigmoid)

| Activation | Test Accuracy | Training Time |
|---|---:|---:|
| ReLU | 57.05% | 9.1 s |
| Sigmoid | 33.29% | 7.6 s |

> ReLU converges significantly faster and achieves higher accuracy due to its non-saturating gradient for positive activations. Sigmoid suffers from vanishing gradients, especially in deeper networks.

---

## Evaluation Metrics

### Baseline CNN

| Metric | Value |
|---|---:|
| Accuracy | 0.6843 |
| Precision | 0.6864 |
| Recall | 0.6843 |
| F1-score | 0.6844 |

### Optimized CNN

| Metric | Value |
|---|---:|
| Accuracy | 0.8334 |
| Precision | 0.8385 |
| Recall | 0.8334 |
| F1-score | 0.8307 |

---

## Per-Class Performance (Optimized CNN)

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Airplane | 0.83 | 0.86 | 0.85 |
| Automobile | 0.87 | 0.97 | 0.92 |
| Bird | 0.88 | 0.67 | 0.76 |
| Cat | 0.79 | 0.63 | 0.70 |
| Deer | 0.87 | 0.76 | 0.81 |
| Dog | 0.79 | 0.77 | 0.78 |
| Frog | 0.70 | 0.96 | 0.81 |
| Horse | 0.90 | 0.87 | 0.88 |
| Ship | 0.90 | 0.91 | 0.91 |
| Truck | 0.86 | 0.93 | 0.90 |
| **Macro Avg** | **0.84** | **0.83** | **0.83** |

---

## Performance Comparison

| Metric | Baseline CNN | Optimized CNN | Improvement |
|---|---:|---:|---|
| Accuracy | 0.6843 | **0.8334** | +14.91% |
| Precision | 0.6864 | **0.8385** | +15.21% |
| Recall | 0.6843 | **0.8334** | +14.91% |
| F1-score | 0.6844 | **0.8307** | +14.63% |
| Architecture | 2-block, Flatten | 3-block, GAP | Deeper + regularized |
| Overfitting | Severe (train ~97%, val ~68%) | Mild (train ≈ val ~83%) | Controlled |
| Epochs to Converge | 20 (fixed, diverges) | ~49 (early stopped) | Adaptive |

---

## Conclusion

The CIFAR-10 dataset was used to study and compare the performance of a baseline CNN against a systematically optimized CNN.

The baseline model, consisting of two convolutional blocks without any regularization or augmentation, achieved a test accuracy of **68.43%** but exhibited severe overfitting — training accuracy climbed to ~97% while validation accuracy plateaued near 68%.

The optimized model solved the issues through: Batch Normalization for training stability, Dropout for co-adaptation prevention, L2 regularization for weight decay, data augmentation for effective dataset expansion, GlobalAveragePooling2D to reduce parameters and overfitting in the classifier head, and adaptive learning rate scheduling via ReduceLROnPlateau. Together, these techniques lifted test accuracy to **83.34%** — an improvement of approximately **14.91 percentage points** — while keeping the training and validation accuracy closely aligned, confirming that overfitting was substantially reduced.

Thus, CNNs with proper regularization significantly outperform both naive CNNs and MLPs on image classification tasks, with architecture depth, filter progression, and regularization strategy being the primary levers for improving generalization.
