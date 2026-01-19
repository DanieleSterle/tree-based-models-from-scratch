import numpy as np
from scipy.special import softmax

import models.decision_tree as dt

def gradient_training(df, labels, column, max_depth, min_sample_split, max_features, learning_rate, n_estimators):
    # Train gradient boosting ensemble for multi-class classification

    logits = np.zeros((labels.shape[0], labels.shape[1]), dtype=float)  # raw scores for each class
    predictions = np.copy(logits)
    trees = []

    for _ in range(n_estimators):
        class_trees = []
        predictions = softmax(predictions, axis=1)  # convert logits to probabilities

        for i in range(labels.shape[1]):
            # Compute residuals (negative gradient)
            residuals = labels[:, i] - predictions[:, i]

            df_copy = df.copy()
            df_copy[column] = residuals
            # Fit decision tree to residuals
            tree = dt.decision_tree(df_copy, column, 0, max_depth, min_sample_split, max_features)
            curr_tree_predictions = dt.predict(tree, df_copy)  

            # Update logits
            logits[:, i] += learning_rate * np.array(curr_tree_predictions)
            class_trees.append(tree)

        trees.append(class_trees)

    return trees

def gradient_predict(df, trees, learning_rate):
    # Predict class probabilities using trained gradient boosting ensemble

    predictions = np.repeat(0, df.shape[0] * len(trees[0])).reshape(df.shape[0], len(trees[0])).astype(float)

    for boosting_round in range(len(trees)):
        for class_idx in range(len(trees[boosting_round])):
            curr_tree_predictions = dt.predict(trees[boosting_round][class_idx], df) 
            predictions[:, class_idx] += learning_rate * np.array(curr_tree_predictions)             

    predictions = softmax(predictions, axis=1)  # convert logits to probabilities

    return predictions