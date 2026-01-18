import pandas as pd
import random as rd

import models.decision_tree as dt

def random_forest(train_df, test_df, column, n_trees, max_depth, min_samples_split, max_features):
    # Trains a random forest of decision trees and returns predictions
    # for the test data using majority voting.

    # Generate random seeds for reproducible bootstrapping
    random_seeds = [rd.randint(0, 10000) for _ in range(n_trees)]
    predictions_df = pd.DataFrame()

    # Train each tree on a bootstrap sample and collect predictions
    for i in range(n_trees):
        sample = train_df.sample(frac=1, replace=True, random_state=random_seeds[i]).reset_index(drop=True)
        tree = dt.decision_tree(sample, column, 0, max_depth, min_samples_split, max_features) 
        predictions_df[f"tree{i + 1}"] = dt.predict(tree, test_df)

    # Return majority vote for each test instance
    return predictions_df.mode(axis=1)[0].tolist()