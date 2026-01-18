import random as rd
import pandas as pd

import utils.criteria as criteria

def decision_tree(df, column, depth, max_depth, min_sample_split, max_features=None):
    # Recursive function to build the decision tree

    # Stop recursion if max depth reached, sample size too small, or pure node
    if depth >= max_depth or len(df) < min_sample_split or df[column].nunique() == 1:
        # Create leaf node with majority class label
        majority_class = df[column].mode()[0]
        return {
            "type": "leaf",
            "prediction": majority_class
        }

    # Select random subset of features (used by Random Forest)
    columns = list(df.columns)
    columns.remove(column)
    random_features = None

    if max_features is not None:
        random_features = rd.sample(columns,  min(max_features, len(columns)))

    # Find best feature and threshold for the split
    best_feature, best_threshold = criteria.best_split(df, random_features, column)
    
    # Split data into left and right branches
    df_under = df[df[best_feature] <= best_threshold]
    df_over = df[df[best_feature] > best_threshold]
    
    # Recursively build subtrees
    left_subtree = decision_tree(df_under, column, depth + 1, max_depth, min_sample_split, max_features)
    right_subtree = decision_tree(df_over, column, depth + 1, max_depth, min_sample_split, max_features)

    # Return decision node
    return {
        "type": "decision",
        "feature": best_feature,
        "threshold": best_threshold,
        "left": left_subtree,
        "right": right_subtree
    }

def predict(tree, samples):
    # Predict class for a single sample or a DataFrame of samples
    
    if isinstance(samples, pd.Series):
        if not criteria.is_leaf(tree):
            feature, threshold = criteria.get_split_info(tree)
            if samples[feature] <= threshold:
                return predict(tree["left"], samples)
            else:
                return predict(tree["right"], samples)
        return tree["prediction"]
    
    predictions = []
    for _, row in samples.iterrows():
        pred = predict(tree, row)
        predictions.append(pred)
        
    return predictions