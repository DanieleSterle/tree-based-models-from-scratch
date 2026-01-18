import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def is_leaf(node):
    # Check if a node is a leaf node
    return node["type"] == "leaf"

def get_split_info(node):
    # Extract splitting feature and threshold from decision node
    return node["feature"], node["threshold"]

def gini_impurity(df, column):
    # Compute Gini impurity for classification labels
    
    unique = df[column].unique()
    probabilities = [(np.sum(df[column] == int(value)) / len(df)) for value in unique]

    return 1 - sum([p**2 for p in probabilities])

def weighted_gini(df_under, df_over, column):
    # Compute weighted Gini impurity for a split

    tot = len(df_under) + len(df_over)
    return (len(df_under) / tot) * gini_impurity(df_under, column) + (len(df_over) / tot) * gini_impurity(df_over, column)

def best_split(df, features, column):
    # Find the best feature and threshold minimizing the chosen gini impurity

    best_feature = None
    best_threshold = None
    best_score = float("inf")
    
    for col in df[features].columns:
        unique = np.sort(df[col].unique())
        
        # Candidate thresholds are midpoints between consecutive values
        thresholds = [((unique[i] + unique[i + 1]) / 2) for i in range(len(unique) - 1)]
        
        for threshold in thresholds:
            df_under = df[df[col] <= threshold]
            df_over = df[df[col] > threshold]
            
            # Skip invalid splits
            if len(df_under) == 0 or len(df_over) == 0:
                continue
            
            score = weighted_gini(df_under, df_over, column)
            
            # Keep best split found so far
            if score < best_score:
                best_score = score
                best_feature = col
                best_threshold = threshold
    
    print(f"Best split: Feature = {best_feature}, Threshold = {best_threshold}, Score = {best_score}")
    return best_feature, best_threshold