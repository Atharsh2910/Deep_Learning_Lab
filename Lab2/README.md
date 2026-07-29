# Experiment 2: Multi-Layer Perceptron (MLP) for Multi-Class Image Classification

## Aim
To implement an MLP using TensorFlow/Keras on the Fashion-MNIST dataset, learning image preprocessing, flattening, model construction, training, evaluation, and automated hyperparameter optimization.

## Dataset Description
**Dataset:** Fashion-MNIST

| Property | Value |
|---|---|
| Training Images | 60,000 |
| Testing Images | 10,000 |
| Classes | 10 |
| Image Size | 28 × 28 |
| Pixel Value Range | [0, 255] |

## Procedure
1. **Dataset Exploration** – Load Fashion-MNIST, print dataset dimensions, display ten sample images, plot class distribution.
2. **Data Preprocessing** – Flatten images (28×28 → 784), normalize pixel values to [0,1], convert labels to one-hot vectors, print tensor shapes before/after preprocessing.
3. **Model Construction** – Build MLP with Input(784) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax); display model summary.
4. **Model Training** – Optimizer: Adam, Loss: Categorical Cross Entropy, Metric: Accuracy; trained for 20 epochs, batch size 32.
5. **Model Evaluation** – Compute Accuracy, Precision, Recall, F1-score, Classification Report, Confusion Matrix.
6. **Hyperparameter Optimization** – RandomizedSearchCV with SciKeras KerasClassifier wrapper and 5-fold cross-validation; retrain and evaluate optimized model, compare with baseline.
7. **Additional Task** – Classification of a XOR gate using manual and Scikit-learn MLP implementations.

## Techniques Used
- Multi-Layer Perceptron (Dense layers with ReLU and Softmax activations)
- Categorical Cross-Entropy Loss
- Adam Optimizer
- Image flattening and normalization
- One-hot encoding of labels
- RandomizedSearchCV (5-fold cross-validation) for hyperparameter optimization
- Dropout regularization (optimized model)

**Hyperparameter Search Space**

| Hyperparameter | Candidate Values |
|---|---|
| Hidden Layers | 1, 2, 3 |
| Hidden Neurons | 32, 64, 128, 256 |
| Learning Rate | 0.1, 0.01, 0.001 |
| Batch Size | 16, 32, 64, 128 |
| Epochs | 10, 20, 30 |
| Optimizer | SGD, Adam, RMSProp |
| Activation Function | ReLU, Tanh, Sigmoid |
| Dropout Rate | 0.0, 0.2, 0.5 |

## Evaluation Metrics

**Baseline Model**

| Metric | Value |
|---|---|
| Accuracy | 0.8790 |
| Precision | 0.8795 |
| Recall | 0.8790 |
| F1-score | 0.8777 |

**Optimized Model**

| Metric | Value |
|---|---|
| Accuracy | 0.8799 |
| Precision | 0.8819 |
| Recall | 0.8799 |
| F1-score | 0.8775 |

**Best Hyperparameters (from RandomizedSearchCV)**

| Hyperparameter | Value |
|---|---|
| Hidden Layers | 1 |
| Hidden Neurons | 256 |
| Learning Rate | 0.001 |
| Batch Size | 16 |
| Optimizer | Adam |
| Activation Function | ReLU |
| Epochs | 20 |
| Dropout | 0.2 |
| Cross-validation Accuracy | 0.8801 |
| Testing Accuracy | 0.8799 |

**Performance Comparison**

| Metric | Baseline | Optimized |
|---|---|---|
| Accuracy | 0.8790 | 0.8799 |
| Precision | 0.8795 | 0.8818 |
| Recall | 0.8790 | 0.8799 |
| F1-score | 0.8776 | 0.8775 |
| Training Time | 84.5 s (20 epochs) | 158.6 s (20 epochs) |

## Conclusion
The Fashion-MNIST dataset was trained and validated using a multi-layer perceptron. The baseline model, built with 784 input neurons, ReLU & Softmax activations, and no optimizer tuning, produced an accuracy of 0.8790. Hyperparameter optimization produced an accuracy of 0.8799, with both baseline and optimized versions yielding nearly identical accuracy, demonstrating the limitations of MLP on image datasets and motivating the use of CNNs for image classification.

On the other hand, the Multi-Layer Perceptron was effective in achieving a perfect classification decision boundary for a XOR gate, which has a non-linear function.

Thus, MLP is useful for classification of non-linear functions effectively. It also classifies images but has a maximum limit on accuracy, beyond which it cannot be optimized.
