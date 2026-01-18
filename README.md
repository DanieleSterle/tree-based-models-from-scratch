# Tree-Based Machine Learning Models From Scratch

A modular, educational Python implementation of core tree-based machine learning algorithms built using only **NumPy**, **Pandas**, and **SciPy**. This project demonstrates the inner workings of Decision Trees, Random Forests, and Gradient Boosting for classification tasks.

## ✨ Features

- **Pure Python Implementation**: Understanding algorithms from the ground up
- **Three Core Models**: Decision Trees, Random Forests, and Gradient Boosting
- **Modular Architecture**: Clean, extensible code structure
- **CLI Interface**: Easy-to-use command-line tool with model-specific subcommands
- **Educational Focus**: Clear code with comprehensive documentation

## 📁 Project Structure

```
TREE_BASED_MODELS/
├── models/
│   ├── decision_tree.py       # Core Decision Tree implementation
│   ├── gradient_boosting.py   # Gradient Boosting with residual learning
│   └── random_forest.py       # Bagging and ensemble voting
├── utils/
│   ├── criteria.py            # Splitting criteria (Gini impurity)
│   ├── data.py                # Data loading and preprocessing
│   ├── utils.py               # CLI parsing and helper functions
│   └── validation.py          # Parameter validation
├── main.py                    # CLI entry point
└── README.md                  # This file
```

## 🚀 Getting Started

### Prerequisites

Ensure you have **Python 3.7+** installed along with the required libraries:

```bash
pip install pandas numpy scipy scikit-learn
```

> **Note**: `scikit-learn` is used only for evaluation metrics (`classification_report`). The core algorithms are implemented from scratch.

## 📊 Data Preparation Requirements

Your input CSV files should follow these guidelines:

| Requirement | Description |
|-------------|-------------|
| **Headers** | First row must contain column names |
| **Target Column** | Must be numeric integers (0, 1, 2, ...) for classification |
| **Features** | All feature columns must be numeric (int or float) |
| **Categorical Data** | Must be pre-encoded using Label Encoding or One-Hot Encoding |
| **Missing Values** | Automatically removed via `df.dropna()` during loading |
| **File Format** | Standard CSV format without commented lines |

## 💻 Usage

The project provides a unified CLI with subcommands for each model type.

### 1. Decision Tree

Train a single decision tree for classification:

```bash
python main.py decision_tree
    -f example.csv
    -c Type
    -r 0.8
    -d 5
    -ms 2
```

### 2. Random Forest

Build an ensemble of decision trees with bootstrap sampling:

```bash
python main.py random_forest
    -f example.csv
    -c Type
    -r 0.8
    -t 10 
    -d 5 
    -ms 2
    -mf 3
```

### 3. Gradient Boosting

Sequential ensemble learning with residual correction for multi-class classification:

```bash
python main.py gradient_boosting
    -f example.csv
    -c Type
    -r 0.8
    -lr 0.1
    -ne 50 
    -d 3
    -ms 5
    -mf 3
```

## 📋 Command-Line Arguments

### Common Arguments (All Models)

| Argument | Short | Type | Required | Description |
|----------|-------|------|----------|-------------|
| `--file_path` | `-f` | str | ✓ | Path to the CSV dataset |
| `--column` | `-c` | str | ✓ | Name of the target column |
| `--ratio` | `-r` | float | ✓ | Train/test split ratio (0.0-1.0) |
| `--max_depth` | `-d` | int | ✓ | Maximum tree depth (0 = unlimited) |
| `--min_samples_split` | `-ms` | int | ✓ | Minimum samples required to split a node |

### Random Forest Specific

| Argument | Short | Type | Required | Description |
|----------|-------|------|----------|-------------|
| `--n_trees` | `-t` | int | ✓ | Number of trees in the forest |
| `--max_features` | `-mf` | int | ✗ | Max features to consider per split (random subset) |

### Gradient Boosting Specific

| Argument | Short | Type | Required | Description |
|----------|-------|------|----------|-------------|
| `--learning_rate` | `-lr` | float | ✓ | Step size shrinkage (0.0-1.0) |
| `--n_estimators` | `-ne` | int | ✓ | Number of boosting rounds |
| `--max_features` | `-mf` | int | ✗ | Max features to consider per split |

## 🔬 Implementation Details

### Decision Tree

**Algorithm:** Greedy recursive binary splitting with CART methodology

**Key Features:**
- **Splitting Criterion:** Gini impurity for classification
- **Threshold Selection:** Tests midpoints between consecutive unique values
- **Node Structure:** 
  - Decision nodes contain feature, threshold, and left/right subtrees
  - Leaf nodes contain majority class prediction
- **Stopping Conditions:** 
  - Maximum depth reached
  - Minimum samples threshold
  - Pure node (single class)


### Random Forest

**Algorithm:** Bootstrap Aggregating (Bagging) with feature randomness

**Key Features:**
- **Bootstrap Sampling:** Each tree trained on random sample with replacement
- **Feature Bagging:** Random subset of features (`max_features`) considered at each split
- **Prediction Aggregation:** 
  - Classification: Majority voting across all trees
  - Uses `pandas.DataFrame.mode()` for efficient voting
- **Variance Reduction:** Decorrelated trees reduce overfitting

**Randomness Sources:**
1. Bootstrap samples (different data subsets)
2. Random feature selection at each split
3. Independent random seeds per tree

### Gradient Boosting

**Algorithm:** Sequential additive modeling with gradient descent

**Key Features:**
- **Multi-class Support:** One-vs-all approach with softmax activation
- **Residual Learning:** Each tree predicts pseudo-residuals (negative gradients)
- **Additive Model:** `F(x) = Σ(learning_rate × tree_prediction)`
- **Training Process:**
  1. Initialize logits to zero
  2. For each boosting round:
     - Convert logits to probabilities via softmax
     - Compute residuals: `labels - predictions`
     - Fit one tree per class to residuals
     - Update logits with scaled tree predictions
- **Regularization:** Learning rate controls contribution of each tree

**Mathematical Foundation:**
```
residual = y_true - softmax(logits)
logits += learning_rate * tree_prediction(residual)
```

## 📊 Example Output

```
Test Accuracy: 0.94

              precision    recall  f1-score   support

           0       0.95      0.91      0.93        22
           1       0.93      0.96      0.94        28

    accuracy                           0.94        50
   macro avg       0.94      0.94      0.94        50
weighted avg       0.94      0.94      0.94        50
```

## 🎯 Algorithm Comparison

| Feature | Decision Tree | Random Forest | Gradient Boosting |
|---------|--------------|---------------|-------------------|
| **Overfitting Risk** | High | Low | Medium |
| **Training Speed** | Fast | Medium | Slow |
| **Interpretability** | High | Low | Low |
| **Typical Accuracy** | Medium | High | High |
| **Parallelization** | N/A | Yes (trees) | No (sequential) |
| **Best Use Case** | Simple patterns | Robust general purpose | Competition-level accuracy |

## 🐛 Validation and Error Handling

The implementation includes comprehensive parameter validation:

- **Decision Tree:** Validates `max_depth ≥ 0`, `min_samples_split ≥ 1`, `0 < ratio < 1`
- **Random Forest:** Additional checks for `n_trees ≥ 1`, `max_features ≥ 1`
- **Gradient Boosting:** Validates `0 < learning_rate ≤ 1`, `n_estimators ≥ 1`

All validations raise descriptive `ValueError` exceptions when constraints are violated.