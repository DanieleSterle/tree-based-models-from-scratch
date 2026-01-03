import numpy as np
import pandas as pd
import scipy as sp
import random as rd
from scipy.special import softmax
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import classification_report

from decision_tree import split_data, predict

def one_hot_encode(labels):
    return pd.get_dummies(labels).values

def best_split(df, features):

    if df is None:
        raise ValueError("Input DataFrame (df) must not be None.")

    if features is None:
        features = [col for col in df.columns if col != "Star type"]
    
    best_feature = None
    best_threshold = None
    best_mse = float("inf")
    
    for col in features:
        if is_numeric_dtype(df[col]):
            unique = np.sort(df[col].unique())
            
            thresholds = [((unique[i] + unique[i + 1]) / 2) for i in range(len(unique) - 1)]
            
            for threshold in thresholds:
                df_under = df[df[col] <= threshold]
                df_over = df[df[col] > threshold]
                
                if len(df_under) == 0 or len(df_over) == 0:
                    continue
                
                mse = weighted_mse(df_under, df_over)

                if mse < best_mse:
                    best_mse = mse
                    best_feature = col
                    best_threshold = threshold
    
    print(f"Best split: Feature = {best_feature}, Threshold = {best_threshold}, MSE = {best_mse}")
    return best_feature, best_threshold

def weighted_mse(left_df, right_df):

    left_mse = np.var(left_df["Star type"]) * len(left_df)
    right_mse = np.var(right_df["Star type"]) * len(right_df)
    total = len(left_df) + len(right_df)

    return (left_mse + right_mse) / total

def decision_tree(df, depth, max_depth, min_sample_split, max_feature):

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

    if depth >= max_depth or len(df) < min_sample_split or df["Star type"].nunique() == 1:
        leaf_prediction = df["Star type"].mean()
        return {
            "type": "leaf",
            "prediction": leaf_prediction
        }
    
    columns = list(df.columns)
    columns.remove("Star type")
    random_features = rd.sample(columns, max_feature)

    best_feature, best_threshold = best_split(df, random_features)
    
    df_under = df[df[best_feature] <= best_threshold]
    df_over = df[df[best_feature] > best_threshold]

    left_subtree = decision_tree(df_under, depth + 1, max_depth, min_sample_split, max_feature)
    right_subtree = decision_tree(df_over, depth + 1, max_depth, min_sample_split, max_feature)
    
    return {
        "type": "decision",
        "feature": best_feature,
        "threshold": best_threshold,
        "left": left_subtree,
        "right": right_subtree
    }

def gradient_training(df, labels, learning_rate, max_feature, n_estimators):

    logits = np.zeros((labels.shape[0], labels.shape[1]), dtype=float)
    predictions = np.copy(logits)
    trees = []

    for _ in range(n_estimators):

        class_trees = []
        predictions = softmax(predictions, axis = 1)

        for i in range(labels.shape[1]):
            residuals = labels[:, i] - predictions[:, i]

            df_copy = df.copy()
            df_copy["Star type"] = residuals
            tree = decision_tree(df_copy, 0, 2, 4, max_feature)
            curr_tree_predictions = predict(tree, df_copy)  

            logits[:, i] += learning_rate * np.array(curr_tree_predictions)
            class_trees.append(tree)

        trees.append(class_trees)

    return trees, learning_rate

def gradient_predict(df, trees, learning_rate):

    predictions = np.repeat(0, df.shape[0] * len(trees[0])).reshape(df.shape[0], len(trees[0])).astype(float)

    for boosting_round in range(len(trees)):
        for class_idx in range(len(trees[boosting_round])):
            curr_tree_predictions = predict(trees[boosting_round][class_idx], df) 
            predictions[:, class_idx] += learning_rate * np.array(curr_tree_predictions)             

    predictions = softmax(predictions, axis=1)

    return predictions

if __name__ == "__main__":

    df = pd.read_csv("stars.csv")
    df = df.dropna()
    df["Star type"] = df["Star type"].astype("Int64")

    train_df, test_df = split_data(df)
    train_labels = one_hot_encode(train_df["Star type"])

    trees, learning_rate = gradient_training(train_df, train_labels, 0.01, 3, 5)

    predictions = np.argmax(gradient_predict(test_df, trees, learning_rate), axis=1)

    accuracy = np.mean(predictions  ==  test_df["Star type"].values)
    print(f"Test Accuracy: {accuracy:.2f}")

    print(classification_report(test_df["Star type"].values, predictions))