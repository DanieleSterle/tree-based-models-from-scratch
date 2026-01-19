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

    # Parse command-line arguments
    args = utils.get_argv()

    # Load dataset and validate target column
    df = data.load_data(args.file_path)
    data.check_column(df, args.column)
    df = df.dropna()

    # Split into train and test sets
    train_df, test_df = data.split_data(df, args.ratio)

    if args.model == "decision_tree":

        # Ensure required hyperparameters are provided
        validation.validate_decision_tree_params(args.max_depth, args.min_samples_split, args.ratio)

        # Train decision tree
        tree = dt.decision_tree(train_df, args.column, 0, args.max_depth, args.min_samples_split, None)

        # Predict on test set
        predictions = dt.predict(tree, test_df)

        # Compute accuracy & Detailed evaluation
        accuracy = np.mean(predictions == test_df[args.column].values)
        print(f"Test Accuracy: {accuracy:.2f}")
        print(classification_report(test_df[args.column], predictions, zero_division=0))

    if args.model == "random_forest":
        # Ensure required hyperparameters are provided
        validation.validate_random_forest_params(args.max_depth, args.min_samples_split, args.max_features, args.n_trees, args.ratio)

        # Train Random Forest and predict
        predictions = rf.random_forest(train_df, test_df, args.column, args.n_trees, args.max_depth, args.min_samples_split, args.max_features)

        # Compute accuracy & Detailed evaluation
        accuracy = np.mean(predictions == test_df[args.column].values)
        print(f"Test Accuracy: {accuracy:.2f}")
        print(classification_report(test_df[args.column], predictions, zero_division=0))

    if args.model == "gradient_boosting":

        # Ensure required hyperparameters are provided
        validation.validate_gradient_boosting_params(args.max_depth, args.min_samples_split, args.max_features, args.n_estimators, args.learning_rate, args.ratio)
        
        # One-hot encode labels for boosting
        train_labels = data.one_hot_encode(train_df[args.column])

        # Train gradient boosting model
        trees = gb.gradient_training(
            train_df, train_labels, args.column,
            args.max_depth, args.min_samples_split,
            args.max_features, args.learning_rate, args.n_estimators
        )

        # Predict class probabilities and take argmax
        predictions = np.argmax(gb.gradient_predict(test_df, trees, args.learning_rate), axis=1)

        # Compute accuracy & Detailed evaluation
        accuracy = np.mean(predictions == test_df[args.column].values)
        print(f"Test Accuracy: {accuracy:.2f}")
        print(classification_report(test_df[args.column].values, predictions, zero_division=0))