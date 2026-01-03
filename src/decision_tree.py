#DF: https://www.kaggle.com/datasets/deepu1109/star-dataset/data
#GINI IMPURITY: https://scientistcafe.com/ids/splitting-criteria

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

import matplotlib.pyplot as plt
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, classification_report

from plotly.subplots import make_subplots

def split_data(df):
    # Shuffle the dataset and split it into 80% train and 20% test

    if df is None:
        raise ValueError("Input DataFrame (df) must not be None.")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(0.8 * len(df))
    
    train_df = df[:split_idx]
    test_df = df[split_idx:]
    
    return train_df, test_df

def gini_impurity(df):
    # Calculate Gini impurity for a DataFrame based on 'Star type' class distribution
    # Gini impurity measures how often a randomly chosen element would be incorrectly labeled

    if df is None:
        raise ValueError("Input DataFrame (df) must not be None.")
    
    unique = df["Star type"].unique()
    probabilities = [(np.sum(df["Star type"] == int(value)) / len(df)) for value in unique]
    
    gini = 1 - sum([p**2 for p in probabilities])  # Formula: 1 - sum of squared probabilities
    return gini

def weighted_gini(df_under, df_over):
    # Compute weighted Gini impurity after a potential split into two groups

    if df_under is None:
        raise ValueError("Input DataFrame 'df_under' must not be None.")

    if df_over is None:
        raise ValueError("Input DataFrame 'df_over' must not be None.")

    tot = len(df_under) + len(df_over)
    return (len(df_under) / tot) * gini_impurity(df_under) + (len(df_over) / tot) * gini_impurity(df_over)

def best_split(df):
    # Find the best feature and threshold to split the data that minimizes weighted Gini impurity
    
    if df is None:
        raise ValueError("Input DataFrame (df) must not be None.")

    best_feature = None
    best_threshold = None
    best_gini = float("inf")  # Initialize with infinity for minimization
    
    for col in df.columns:
        if is_numeric_dtype(df[col]):
            unique = np.sort(df[col].unique())  # Sort unique values to find midpoints
            
            # Calculate candidate thresholds as midpoints between consecutive unique values
            thresholds = [((unique[i] + unique[i + 1]) / 2) for i in range(len(unique) - 1)]
            
            for threshold in thresholds:
                df_under = df[df[col] <= threshold]
                df_over = df[df[col] > threshold]
                
                # Skip splits that don't divide the data
                if len(df_under) == 0 or len(df_over) == 0:
                    continue
                
                gini = weighted_gini(df_under, df_over)
                
                # Update best split if this split yields lower weighted Gini impurity
                if gini < best_gini:
                    best_gini = gini
                    best_feature = col
                    best_threshold = threshold
    
    print(f"Best split: Feature = {best_feature}, Threshold = {best_threshold}, Gini = {best_gini}")
    return best_feature, best_threshold

def decision_tree(df, depth, max_depth, min_sample_split):
    # Recursive function to build the decision tree
    
    if df is None:
        raise ValueError("Input DataFrame (df) must not be None.")

    if depth < 0:
        raise ValueError("depth must be 0 or greater.")

    if max_depth < 0:
        raise ValueError("max_depth must be 0 or greater.")

    if min_sample_split < 1:
        raise ValueError("min_sample_split must be at least 1.")

    # Stop recursion if max depth reached, sample size too small, or pure node
    if depth >= max_depth or len(df) < min_sample_split or df["Star type"].nunique() == 1:
        # Create leaf node with majority class label
        majority_class = df["Star type"].mode()[0]
        return {
            "type": "leaf",
            "prediction": majority_class
        }

    # Find best feature and threshold to split current node
    best_feature, best_threshold = best_split(df)
    
    # Split data into left and right branches
    df_under = df[df[best_feature] <= best_threshold]
    df_over = df[df[best_feature] > best_threshold]
    
    # Recursively build left and right subtrees
    left_subtree = decision_tree(df_under, depth + 1, max_depth, min_sample_split)
    right_subtree = decision_tree(df_over, depth + 1, max_depth, min_sample_split)
    
    # Return decision node with split info and child branches
    return {
        "type": "decision",
        "feature": best_feature,
        "threshold": best_threshold,
        "left": left_subtree,
        "right": right_subtree
    }

def predict(tree, samples):
    
    # If input is a single sample (Series)
    if isinstance(samples, pd.Series):
        if not is_leaf(tree):
            feature, threshold = get_split_info(tree)
            # Traverse left or right subtree depending on feature value
            if samples[feature] <= threshold:
                return predict(tree["left"], samples)
            else:
                return predict(tree["right"], samples)
        # Leaf node: return prediction
        return tree["prediction"]
    
    # If input is a DataFrame of samples
    predictions = []
    for _, row in samples.iterrows():
        pred = predict(tree, row)
        predictions.append(pred)
    return predictions

def is_leaf(node):
    # Check if a node is a leaf node
    return node["type"] == "leaf"

def get_split_info(node):
    # Extract splitting feature and threshold from decision node
    return node["feature"], node["threshold"]

if __name__  ==  "__main__":
    # Load dataset
    df = pd.read_csv("stars.csv")
    df = df.dropna()  # Remove missing values

    # Split data into training and testing sets
    train_df, test_df = split_data(df)

    # Build decision tree with specified max depth and minimum samples per split
    tree = decision_tree(train_df, 0, 5, 10)

    # Predict on test set
    predictions = predict(tree, test_df)

    # Calculate and print test accuracy
    correct = sum(1 for true, pred in zip(test_df["Star type"], predictions) if true == pred)
    accuracy = correct / len(test_df)
    print(f"Test Accuracy: {accuracy:.2f}")

    # Print detailed classification report
    print(classification_report(test_df["Star type"], predictions))

    # Sort the entire dataset by temperature for visualization
    df = df.sort_values("Temperature (K)", ascending=True)

    # Get unique star types for color mapping
    star_types = df["Star type"].unique()
    colors = plt.cm.get_cmap("tab10", len(star_types))  # Use colormap with enough colors

    plt.figure(figsize=(8, 5))

    # Plot data points for each star type with different colors
    for i, star_type in enumerate(star_types):
        subset = df[df["Star type"] == star_type]
        plt.scatter(
            subset["Temperature (K)"],
            subset["Absolute magnitude(Mv)"],
            label=f"Type {star_type}",
            color=colors(i),
            s=30,
            alpha=0.7
        )

    plt.xlabel("Temperature (K)")
    plt.ylabel("Absolute Magnitude (Mv)")
    plt.title("Hertzsprung–Russell Diagram (Color-coded by Star Type)")

    # Invert axes to match astronomical convention (hot stars left, bright stars top)
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()