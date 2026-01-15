import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

import utils.data as data
import utils.utils as utils
from utils.criteria import CRITERIA
import models.decision_tree as dt
import models.random_forest as rf
import models.gradient_boosting as gb

if __name__ == "__main__":

    # Read all command-line arguments
    model, file_path, column, criterion, ratio, max_depth, min_samples_split, n_trees, max_features, learning_rate, n_estimators = utils.get_argv() 

    if ratio <= 0.0 or ratio >= 1.0:
        raise ValueError("Ratio must be between 0 and 1.")

    # Validate criterion
    if criterion not in CRITERIA:
        raise ValueError(
            f"Unknown criterion '{criterion}'. "
            f"Available: {list(CRITERIA.keys())}"
        )

    criterion_fn = CRITERIA[criterion]

    # Load dataset and validate target column
    df = data.load_data(file_path)
    data.check_column(df, column)
    df = df.dropna()

    # Split into train and test sets
    train_df, test_df = data.split_data(df, ratio)

    if model == "decision_tree":

        # Ensure required hyperparameters are provided
        if max_depth is None or min_samples_split is None:
            raise ValueError("For decision_tree model, depth and min_samples_split arguments are required.")

        # Train decision tree
        tree = dt.decision_tree(train_df, column, criterion_fn, 0, max_depth, min_samples_split, None)

        # Predict on test set
        predictions = dt.predict(tree, test_df)

        # Compute accuracy & Detailed evaluation
        correct = sum(1 for true, pred in zip(test_df[column], predictions) if true == pred)
        accuracy = correct / len(test_df)
        print(f"Test Accuracy: {accuracy:.2f}")
        print(classification_report(test_df[column], predictions))

    if model == "random_forest":

        # Ensure required hyperparameters are provided
        if n_trees is None or max_features is None or max_depth is None or min_samples_split is None:
            raise ValueError("For random_forest model, n_trees, max_features, max_depth, and min_samples_split arguments are required.")

        # Train Random Forest and predict
        predictions = rf.random_forest(train_df, test_df, column, criterion_fn, n_trees, max_depth, min_samples_split, max_features)

        # Model evaluation
        print(classification_report(test_df[column], predictions))

    if model == "gradient_boosting":

        # Ensure required hyperparameters are provided
        if learning_rate is None or n_estimators is None or max_features is None or max_depth is None or min_samples_split is None:
            raise ValueError("For gradient_boosting model, learning_rate, n_estimators, max_features, max_depth, and min_samples_split arguments are required.")
        
        # One-hot encode labels for boosting
        train_labels = utils.one_hot_encode(train_df[column])

        # Train gradient boosting model
        trees, learning_rate = gb.gradient_training(
            train_df, train_labels, column,
            criterion_fn, max_depth, min_samples_split,
            max_features, learning_rate, n_estimators
        )

        # Predict class probabilities and take argmax
        predictions = np.argmax(gb.gradient_predict(test_df, trees, learning_rate), axis=1)

        # Compute accuracy & Detailed evaluation
        accuracy = np.mean(predictions == test_df[column].values)
        print(f"Test Accuracy: {accuracy:.2f}")
        print(classification_report(test_df[column].values, predictions))