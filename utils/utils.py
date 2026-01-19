import argparse

def get_argv():
    # Argument parser with subcommands for each model
    parser = argparse.ArgumentParser(description="Train ML models")
    subparsers = parser.add_subparsers(dest="model", required=True)

    # Common arguments shared by all models
    def add_common_args(p):
        p.add_argument("-f", "--file_path", dest="file_path", required=True, help="Path to .csv file")
        p.add_argument("-c", "--column", dest="column", required=True, help="Target column")
        p.add_argument("-r", "--ratio", dest="ratio", required=True, type=float, help="Train/test split ratio")

    # ---------------- Decision Tree ----------------
    dt_parser = subparsers.add_parser("decision_tree", help="Decision Tree model")
    add_common_args(dt_parser)
    dt_parser.add_argument("-d", "--max_depth", dest="max_depth", required=True, type=int, help="Max depth")
    dt_parser.add_argument("-ms", "--min_samples_split", dest="min_samples_split", required=True, type=int, help="Min samples to split")

    # ---------------- Random Forest ----------------
    rf_parser = subparsers.add_parser("random_forest", help="Random Forest model")
    add_common_args(rf_parser)
    rf_parser.add_argument("-d", "--max_depth", dest="max_depth", required=True, type=int, help="Max depth")
    rf_parser.add_argument("-ms", "--min_samples_split", dest="min_samples_split", required=True, type=int, help="Min samples to split")
    rf_parser.add_argument("-mf", "--max_features", dest="max_features", required=False, type=int, help="Max features")
    rf_parser.add_argument("-t", "--n_trees", dest="n_trees", required=True, type=int, help="Number of trees")

    # ---------------- Gradient Boosting ----------------
    gb_parser = subparsers.add_parser("gradient_boosting", help="Gradient Boosting model")
    add_common_args(gb_parser)
    gb_parser.add_argument("-d", "--max_depth", dest="max_depth", required=True, type=int, help="Max depth")
    gb_parser.add_argument("-ms", "--min_samples_split", dest="min_samples_split", required=True, type=int, help="Min samples to split")
    gb_parser.add_argument("-mf", "--max_features", dest="max_features", required=False, type=int, help="Max features")
    gb_parser.add_argument("-ne", "--n_estimators", dest="n_estimators", required=True, type=int, help="Number of estimators")
    gb_parser.add_argument("-lr", "--learning_rate", dest="learning_rate", required=True, type=float, help="Learning rate")

    return parser.parse_args()