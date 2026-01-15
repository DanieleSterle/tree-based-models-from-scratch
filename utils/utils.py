import argparse
import pandas as pd

def get_argv():
    # Argument parser with subcommands for each model
    parser = argparse.ArgumentParser(description="Train ML models")
    subparsers = parser.add_subparsers(dest="model", required=True)

    # Common arguments shared by all models
    def add_common_args(p):
        p.add_argument("-f", dest="file_path", required=True, help="Path to .csv file")
        p.add_argument("-c", dest="column", required=True, help="Target column")
        p.add_argument("-r", "--ratio", dest="ratio", required=True, type=float, help="Train/test split ratio")

    # ---------------- Decision Tree ----------------
    dt_parser = subparsers.add_parser("decision_tree", help="Decision Tree model")
    add_common_args(dt_parser)
    dt_parser.add_argument("-d", "--max_depth", dest="max_depth", required=True, type=int, help="Max depth")
    dt_parser.add_argument("-ms", "--min_samples_split", dest="min_samples_split", required=True, type=int, help="Min samples to split")
    dt_parser.add_argument("-cr", "--criterion", dest="criterion", required=True, help="Criterion for splitting")

    # ---------------- Random Forest ----------------
    rf_parser = subparsers.add_parser("random_forest", help="Random Forest model")
    add_common_args(rf_parser)
    rf_parser.add_argument("-t", "--n_trees", dest="n_trees", required=True, type=int, help="Number of trees")
    rf_parser.add_argument("-mf", "--max_features", dest="max_features", required=True, type=int, help="Max features")
    rf_parser.add_argument("-d", "--max_depth", dest="max_depth", required=True, type=int, help="Max depth")
    rf_parser.add_argument("-ms", "--min_samples_split", dest="min_samples_split", required=True, type=int, help="Min samples to split")
    rf_parser.add_argument("-cr", "--criterion", dest="criterion", required=True, help="Criterion for splitting")

    # ---------------- Gradient Boosting ----------------
    gb_parser = subparsers.add_parser("gradient_boosting", help="Gradient Boosting model")
    add_common_args(gb_parser)
    gb_parser.add_argument("-lr", "--learning_rate", dest="learning_rate", required=True, type=float, help="Learning rate")
    gb_parser.add_argument("-ne", "--n_estimators", dest="n_estimators", required=True, type=int, help="Number of estimators")
    gb_parser.add_argument("-mf", "--max_features", dest="max_features", required=True, type=int, help="Max features")
    gb_parser.add_argument("-d", "--max_depth", dest="max_depth", required=True, type=int, help="Max depth")
    gb_parser.add_argument("-ms", "--min_samples_split", dest="min_samples_split", required=True, type=int, help="Min samples to split")
    gb_parser.add_argument("-cr", "--criterion", dest="criterion", required=True, help="Criterion for splitting")

    args = parser.parse_args()

    # Extract only arguments relevant to the selected model
    model = args.model
    file_path = args.file_path
    column = args.column

    max_depth = args.max_depth if hasattr(args, "max_depth") else None
    ratio = args.ratio if hasattr(args, "ratio") else None
    criterion = args.criterion if hasattr(args, "criterion") else None
    min_samples_split = args.min_samples_split if hasattr(args, "min_samples_split") else None
    n_trees = args.n_trees if hasattr(args, "n_trees") else None
    max_features = args.max_features if hasattr(args, "max_features") else None
    learning_rate = args.learning_rate if hasattr(args, "learning_rate") else None
    n_estimators = args.n_estimators if hasattr(args, "n_estimators") else None

    return model, file_path, column, criterion, ratio, max_depth, min_samples_split, n_trees, max_features, learning_rate, n_estimators

def one_hot_encode(labels):
    # Convert class labels to one-hot encoded matrix
    return pd.get_dummies(labels).values