import pandas as pd
from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split


def mean_centering(user_item_matrix: pd.DataFrame) -> Tuple[pd.DataFrame, np.array]:
    means = user_item_matrix.mean(axis=1)
    return user_item_matrix.sub(means, axis=0), means.to_numpy()


def train_test_data_from_path(path: str):
    # Load ratings file
    df = pd.read_csv(path, delimiter="\t", header=None,
                     names=['user_id', 'item_id', 'rating', 'timestamp'])
    # Remove unused column
    del df["timestamp"]
    # pivot ratings list into ratings matrix
    train, test = train_test_split(df, test_size=0.2)

    df_ratings = train.pivot(index="user_id", columns="item_id", values="rating")
    return df_ratings, test
