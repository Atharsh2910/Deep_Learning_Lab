# Experiment 1: Single Layer Perceptron for Binary Classification

## Aim
To understand the concept of an artificial neuron and implement a Single Layer Perceptron from scratch for binary classification, using the perceptron learning algorithm, activation functions, learning process visualization, and evaluation on a real-world dataset.

## Dataset Description
**Dataset:** Banknote Authentication Dataset
**Source:** UCI Machine Learning Repository

| Property | Value |
|---|---|
| Instances | 1372 |
| Features | 4 Numerical Features |
| Classes | 2 |
| Missing Values | None |
| Task | Binary Classification |

**Features:** Variance, Skewness, Curtosis, Entropy

**Target Class:**
- 0 – Authentic Banknote
- 1 – Forged Banknote

## Procedure
1. **Dataset Exploration** – Load dataset, display first five samples, determine dimensions, identify missing values, display descriptive statistics.
2. **Exploratory Data Analysis** – Generate histograms, correlation heatmap, scatter plot, boxplots.
3. **Data Preprocessing** – Normalize all numerical features; split into Training (80%) and Testing (20%).
4. **Perceptron Implementation (from scratch)** – Weight initialization, bias initialization, step activation function, forward propagation, perceptron learning rule.
5. **Model Training** – Track epoch number, misclassified samples, updated weights, updated bias.
6. **Model Evaluation** – Compute Accuracy, Precision, Recall, F1-score, Confusion Matrix.
7. **Additional Tasks** – Comparison of Step vs Sigmoid activation, comparison with Scikit-learn Perceptron, learning rate study (0.001, 0.01, 0.1), XOR limitation demonstration, effect of feature normalization, and classification of OR/AND/NOT logic gates (Week 3).

## Techniques Used
- Single Layer Perceptron implemented from scratch
- Step Activation Function
- Perceptron Learning Rule (weight and bias update)
- Feature normalization
- Train-test split (80:20)
- Comparison with Scikit-learn's Perceptron

## Evaluation Metrics
| Metric | Value |
|---|---|
| Accuracy | 0.9891 |
| Precision | 0.9917 |
| Recall | 0.9836 |
| F1-score | 0.9877 |

**Confusion Matrix**

|            | Predicted −1 | Predicted +1 |
|---|---|---|
| **Actual −1** | 152 (TN) | 1 (FP) |
| **Actual +1** | 2 (FN) | 120 (TP) |

**Manual vs Scikit-learn Perceptron**

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Manual Perceptron | 0.9891 | 0.9917 | 0.9836 | 0.9877 |
| Scikit-learn Perceptron | 0.9709 | 0.9385 | 1.0000 | 0.9683 |

## Conclusion
- The model achieved **98.91% accuracy**, **99.17% precision**, **98.36% recall**, and an **F1-score of 98.77%** on the test set, correctly classifying 272 of 275 test samples.
- Both the manual and Scikit-learn perceptron achieved almost similar accuracy (98.91% vs. 97.09%).
- This shows that a Single Layer Perceptron can be used for binary classification efficiently.
- The additional tasks of classifying logic gates show that a SLP can effectively classify OR, AND, and NOT gates (linear functions), but fails to classify the XOR gate (non-linear function), proving that SLP is limited to a linear decision boundary, y = mx + c.
