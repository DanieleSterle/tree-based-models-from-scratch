# Star Type Classification using Custom Decision Tree, Random Forest & Gradient Boosting

##  Overview

This project implements **Decision Tree**, **Random Forest**, and a simplified **Gradient Boosting Classifier** from scratch—using only core Python libraries. The objective is to classify **star types** based on astrophysical properties such as temperature, luminosity, radius, and spectral class.

---

##  Dataset

We use the [Star Dataset from Kaggle](https://www.kaggle.com/datasets/deepu1109/star-dataset/data), which includes the following features:

- **Temperature (K)** – Surface temperature of the star  
- **Luminosity (L/Lo)** – Luminosity in solar units  
- **Radius (R/Ro)** – Radius in solar radii  
- **Absolute magnitude (Mv)** – Brightness of the star  
- **Star color** – Color (categorical)  
- **Spectral Class** – Spectral type (categorical)  
- **Star type** – Target variable (integer from 0 to 5)

---

##  Features

- Manual implementation of Gini impurity
- Recursive decision tree building
- Bagging and feature sampling for random forest
- Custom gradient boosting classifier using softmax and MSE
- One-hot encoding for multi-class labels
- Evaluation using precision, recall, F1-score
- Hertzsprung–Russell Diagram for data visualization

---

##  Decision Tree Classifier

### Gini Impurity (for classification)

\[
Gini(D) = 1 - \sum p_i^2
\]

Where \( p_i \) is the probability of class \( i \) in dataset \( D \).

### Stopping Criteria

- Max depth reached  
- Minimum sample size for split violated  
- Node is pure (single class)

---

##  Random Forest Classifier

- Trains multiple decision trees on **bootstrap samples**
- For each split, selects a **random subset of features**
- Predictions made by **majority vote**

---

## Gradient Boosting Classifier

A custom multi-class gradient boosting model is implemented using softmax and MSE loss.

###  How It Works

1. **One-hot Encode Labels**  
   Multi-class labels are converted to binary matrices.

2. **Compute Residuals**  
   Residual = true_label - predicted_probability (per class)

3. **Train Regression Trees**  
   One tree per class is trained on residuals using **Mean Squared Error (MSE)**.

4. **Boosting Rounds**  
   Repeats for a fixed number of iterations (`n_estimators`), improving predictions iteratively.

5. **Final Prediction**  
   Logits are passed through softmax to generate class probabilities.

---

##  Example Output
    Best split: Feature = Luminosity(L/Lo), Threshold = 0.875, MSE = 0.1247

    Test Accuracy: 0.88
                precision    recall  f1-score   support

            0       1.00      1.00      1.00        11
            1       0.92      0.86      0.89         7
            2       1.00      1.00      1.00         8
            3       0.80      0.80      0.80         5
            4       0.50      0.33      0.40         3
            5       0.67      0.67      0.67         3

    accuracy                            0.88        37
    macro avg       0.82       0.78     0.79        37
    weighted avg    0.88       0.88     0.88        37

---

## Visualization: Hertzsprung–Russell Diagram

A scatter plot is created with:

- **X-axis**: Temperature (reversed — hotter stars on the left)
- **Y-axis**: Absolute Magnitude (reversed — brighter stars on top)
- **Color**: Encodes the star type