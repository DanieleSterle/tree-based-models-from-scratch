import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def is_leaf(node):
    # Check if a node is a leaf node
    return node["type"] == "leaf"

def get_split_info(node):
    # Extract splitting feature and threshold from decision node
    return node["feature"], node["threshold"]

def weighted_mse(left_df, right_df, column):
    # Compute weighted Mean Squared Error for regression splits
    left_mse = np.var(left_df[column]) * len(left_df)
    right_mse = np.var(right_df[column]) * len(right_df)
    total = len(left_df) + len(right_df)

    return (left_mse + right_mse) / total

def gini_impurity(df, column):
    # Compute Gini impurity for classification labels

    if df is None:
        raise ValueError("Input DataFrame (df) must not be None.")
    
    unique = df[column].unique()
    probabilities = [(np.sum(df[column] == int(value)) / len(df)) for value in unique]

    return 1 - sum([p**2 for p in probabilities])

def weighted_gini(df_under, df_over, column):
    # Compute weighted Gini impurity for a split

    if df_under is None:
        raise ValueError("Input DataFrame 'df_under' must not be None.")

    if df_over is None:
        raise ValueError("Input DataFrame 'df_over' must not be None.")

    tot = len(df_under) + len(df_over)
    return (len(df_under) / tot) * gini_impurity(df_under, column) + (len(df_over) / tot) * gini_impurity(df_over, column)

CRITERIA = {
    "gini": weighted_gini,
    "mse": weighted_mse
}

def best_split(df, features, column, criterion):
    # Find the best feature and threshold minimizing the chosen split criterion
    
    if df is None:
        raise ValueError("Input DataFrame (df) must not be None.")
    
    if features is None or len(features) == 0:
        features = [col for col in df.columns if col != column]

    best_feature = None
    best_threshold = None
    best_score = float("inf")
    
    for col in df[features].columns:
        if is_numeric_dtype(df[col]):
            unique = np.sort(df[col].unique())
            
            # Candidate thresholds are midpoints between consecutive values
            thresholds = [((unique[i] + unique[i + 1]) / 2) for i in range(len(unique) - 1)]
            
            for threshold in thresholds:
                df_under = df[df[col] <= threshold]
                df_over = df[df[col] > threshold]
                
                # Skip invalid splits
                if len(df_under) == 0 or len(df_over) == 0:
                    continue
                
                score = criterion(df_under, df_over, column)
                
                # Keep best split found so far
                if score < best_score:
                    best_score = score
                    best_feature = col
                    best_threshold = threshold
    
    print(f"Best split: Feature = {best_feature}, Threshold = {best_threshold}, Score = {best_score}")
    return best_feature, best_threshold