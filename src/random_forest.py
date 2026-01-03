import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import random as rd

import matplotlib.pyplot as plt
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, classification_report

from plotly.subplots import make_subplots

from decision_tree import split_data, predict, weighted_gini, gini_impurity, get_split_info, is_leaf

def best_split(df, features):
    """
    Find the best feature and threshold to split the data
    based on the minimum weighted Gini impurity.
    """
    if df is None:
        raise ValueError("Input DataFrame (df) must not be None.")

    # If no features are passed, use all except target
    if features is None:
        features = [col for col in df.columns if col != "Star type"]
    
    best_feature = None
    best_threshold = None
    best_gini = float("inf")  # Initialize with infinity for minimization
    
    for col in features:
        if is_numeric_dtype(df[col]):
            unique = np.sort(df[col].unique())  # Get sorted unique values
            
            # Compute candidate thresholds between each pair of consecutive values
            thresholds = [((unique[i] + unique[i + 1]) / 2) for i in range(len(unique) - 1)]
            
            for threshold in thresholds:
                df_under = df[df[col] <= threshold]
                df_over = df[df[col] > threshold]
                
                # Skip invalid splits
                if len(df_under) == 0 or len(df_over) == 0:
                    continue
                
                # Calculate weighted Gini for this split
                gini = weighted_gini(df_under, df_over)
                
                # Keep track of the best split
                if gini < best_gini:
                    best_gini = gini
                    best_feature = col
                    best_threshold = threshold
    
    print(f"Best split: Feature = {best_feature}, Threshold = {best_threshold}, Gini = {best_gini}")
    return best_feature, best_threshold

def decision_tree(df, depth, max_depth, min_sample_split, max_feature):
    """
    Recursively builds a decision tree based on Gini impurity.
    """
    # Input validation
    if df is None:
        raise ValueError("Input DataFrame (df) must not be None.")
    if depth < 0:
        raise ValueError("depth must be 0 or greater.")
    if max_depth < 0:
        raise ValueError("max_depth must be 0 or greater.")
    if min_sample_split < 1:
        raise ValueError("min_sample_split must be at least 1.")
    if max_feature is not None and max_feature >= len(df.columns):
        raise ValueError(f"max_feature ({max_feature}) must be less than the number of columns ({len(df.columns)}) in the DataFrame.")

    # Base cases for recursion
    if depth >= max_depth or len(df) < min_sample_split or df["Star type"].nunique() == 1:
        majority_class = df["Star type"].mode()[0]
        return {
            "type": "leaf",
            "prediction": majority_class
        }
    
    # Select random subset of features for splitting (as in Random Forest)
    columns = list(df.columns)
    columns.remove("Star type")
    random_features = rd.sample(columns, max_feature)

    # Find the best split among selected features
    best_feature, best_threshold = best_split(df, random_features)
    
    # Split dataset based on the best threshold
    df_under = df[df[best_feature] <= best_threshold]
    df_over = df[df[best_feature] > best_threshold]

    # Recursively build left and right subtrees
    left_subtree = decision_tree(df_under, depth + 1, max_depth, min_sample_split, max_feature)
    right_subtree = decision_tree(df_over, depth + 1, max_depth, min_sample_split, max_feature)
    
    # Return current decision node
    return {
        "type": "decision",
        "feature": best_feature,
        "threshold": best_threshold,
        "left": left_subtree,
        "right": right_subtree
    }

def random_forest(train_df, test_df, n_trees):
    """
    Trains a random forest of decision trees and returns predictions
    for the test data using majority voting.
    """
    # Input validation
    if n_trees < 1:
        raise ValueError("n_trees must be at least 1.")
    if train_df.empty:
        raise ValueError("Training DataFrame is empty.")
    if test_df.empty:
        raise ValueError("Test DataFrame is empty.")
    if "Star type" not in train_df.columns:
        raise ValueError("'Star type' column is missing in the training DataFrame.")
    
    # Generate random seeds for reproducible bootstrapping
    random_seeds = [rd.randint(0, 10000) for _ in range(n_trees)]
    predictions_df = pd.DataFrame()

    # Train each tree on a bootstrap sample and collect predictions
    for i in range(n_trees):
        sample = train_df.sample(frac=1, replace=True, random_state=random_seeds[i]).reset_index(drop=True)
        tree = decision_tree(sample, 0, 5, 10, 3)  # depth=0, max_depth=5, min_sample_split=10, max_feature=3
        predictions_df[f"tree{i + 1}"] = predict(tree, test_df)

    # Return majority vote for each test instance
    return predictions_df.mode(axis=1)[0].tolist()

if __name__ == "__main__":
    # Load and prepare the dataset
    df = pd.read_csv("stars.csv")
    df = df.dropna()  # Drop any rows with missing values

    # Split into training and testing sets
    train_df, test_df = split_data(df)

    # Train Random Forest with 2 trees and make predictions
    predictions = random_forest(train_df, test_df, 2)

    # Evaluate model performance
    print(classification_report(test_df["Star type"], predictions))
