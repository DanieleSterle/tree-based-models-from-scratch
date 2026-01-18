def validate_decision_tree_params(max_depth, min_samples_split, ratio):
    # Validate decision tree specific parameters.

    if max_depth < 0:
        raise ValueError("max_depth must be 0 or greater.")

    if min_samples_split < 1:
        raise ValueError("min_sample_split must be at least 1.")
    
    if ratio <= 0.0 or ratio >= 1.0:
        raise ValueError("Ratio must be between 0 and 1.")


def validate_random_forest_params(max_depth, min_samples_split, max_features, n_trees, ratio):
    # Validate Random Forest specific parameters.

    if max_depth < 0:
        raise ValueError("max_depth must be 0 or greater.")

    if min_samples_split < 1:
        raise ValueError("min_samples_split must be at least 1.")
    
    if max_features is not None and max_features < 1:
        raise ValueError("max_features must be at least 1.")

    if n_trees < 1:
        raise ValueError("n_trees must be at least 1.")
    
    if ratio <= 0.0 or ratio >= 1.0:
        raise ValueError("Ratio must be between 0 and 1.")


def validate_gradient_boosting_params(max_depth, min_samples_split, max_features, n_estimators, learning_rate, ratio):
    # Validate Gradient Boosting specific parameters.

    if max_depth < 0:
        raise ValueError("max_depth must be 0 or greater.")

    if min_samples_split < 1:
        raise ValueError("min_samples_split must be at least 1.")

    if max_features is not None and max_features < 1:
        raise ValueError("max_features must be at least 1.")

    if n_estimators < 1:
        raise ValueError("n_estimators must be at least 1.")

    if not 0.0 < learning_rate <= 1.0:
        raise ValueError("learning_rate must be between 0 and 1.")
    
    if ratio <= 0.0 or ratio >= 1.0:
        raise ValueError("Ratio must be between 0 and 1.")