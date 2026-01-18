import pandas as pd
from pandas.api.types import is_numeric_dtype

def load_data(file_path):
    # Load CSV file and ignore commented lines
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError("File not found.")

    return df

def split_data(df, ratio):
    # Shuffle dataset and split into train and test sets
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(ratio * len(df))

    train_df = df[:split_idx]
    test_df = df[split_idx:]

    return train_df, test_df

def check_column(df, name_column):
    # Validate that target column exists and is numeric
    if name_column not in df.columns:
        raise ValueError(f"Column '{name_column}' does not exist in the DataFrame.")

    if not is_numeric_dtype(df[name_column]):
        raise ValueError(f"Column '{name_column}' must be numeric.")
    
def one_hot_encode(labels):
    # Convert class labels to one-hot encoded matrix
    return pd.get_dummies(labels).values