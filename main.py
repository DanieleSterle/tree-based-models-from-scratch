import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

import utils.data as data
import utils.utils as utils
import utils.validation as validation
import models.decision_tree as dt
import models.random_forest as rf
import models.gradient_boosting as gb

if __name__ == "__main__":

    # Read all command-line arguments
    model, file_path, column, ratio, max_depth, min_samples_split, max_features, n_trees, n_estimators, learning_rate = utils.get_argv() 

    # Load dataset and validate target column
    df = data.load_data(file_path)
    data.check_column(df, column)
    df = df.dropna()

    # Split into train and test sets
    train_df, test_df = data.split_data(df, ratio)

    if model == "decision_tree":

        # Ensure required hyperparameters are provided
        validation.validate_decision_tree_params(max_depth, min_samples_split, ratio)

        # Train decision tree
        tree = dt.decision_tree(train_df, column, 0, max_depth, min_samples_split, None)

        # Predict on test set
        predictions = dt.predict(tree, test_df)

        # Compute accuracy & Detailed evaluation
        accuracy = np.mean(predictions == test_df[column].values)
        print(f"Test Accuracy: {accuracy:.2f}")
        print(classification_report(test_df[column], predictions, zero_division=0))

    if model == "random_forest":

        # Ensure required hyperparameters are provided
        validation.validate_random_forest_params(max_depth, min_samples_split, max_features, n_trees, ratio)

        # Train Random Forest and predict
        predictions = rf.random_forest(train_df, test_df, column, n_trees, max_depth, min_samples_split, max_features)

        # Compute accuracy & Detailed evaluation
        accuracy = np.mean(predictions == test_df[column].values)
        print(f"Test Accuracy: {accuracy:.2f}")
        print(classification_report(test_df[column], predictions, zero_division=0))

    if model == "gradient_boosting":

        # Ensure required hyperparameters are provided
        validation.validate_gradient_boosting_params(max_depth, min_samples_split, max_features, n_estimators, learning_rate, ratio)
        
        # One-hot encode labels for boosting
        train_labels = data.one_hot_encode(train_df[column])

        # Train gradient boosting model
        trees = gb.gradient_training(
            train_df, train_labels, column,
            max_depth, min_samples_split,
            max_features, learning_rate, n_estimators
        )

        # Predict class probabilities and take argmax
        predictions = np.argmax(gb.gradient_predict(test_df, trees, learning_rate), axis=1)

        # Compute accuracy & Detailed evaluation
        accuracy = np.mean(predictions == test_df[column].values)
        print(f"Test Accuracy: {accuracy:.2f}")
        print(classification_report(test_df[column].values, predictions, zero_division=0))