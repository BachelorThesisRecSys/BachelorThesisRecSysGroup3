from typing import Tuple
import numpy as np
import pandas as pd
import scipy.linalg


def k_nearest_symmetric(similarity_matrix:pd.DataFrame, K: int) -> Tuple[pd.DataFrame, np.array]:
    # for a row, find K - column_nonzero most similar values
    k_nearest = similarity_matrix.copy(deep=True) * np.nan

    for idx, row in similarity_matrix.iterrows():
        used = k_nearest[idx].count() #number of used entries
        if used >= K:
            continue
        # do not select the rows that already have 10 column entries
        mask_done = k_nearest.count() < K
        row_upper = row[idx:]

        k_nearest.loc[idx] = row_upper[mask_done].nlargest(K - used, keep='first')

    k_nearest = k_nearest.fillna(0)
    k_nearest_sym = k_nearest + k_nearest.T
    degree = np.zeros(k_nearest_sym.shape)
    np.fill_diagonal(degree, k_nearest_sym.sum())
    return k_nearest_sym, degree


def tikhonov(adjacency:pd.DataFrame, degree:np.ndarray, ratings:pd.DataFrame, means, mu:float):
    L = degree - adjacency.to_numpy()
    X_star = np.linalg.inv(np.identity(degree.shape[0]) + mu * L) @ ratings
    X_star = X_star.add(means.values, axis=0)
    X_star.index = adjacency.index
    return X_star


def sobolev(adjacency:pd.DataFrame, degree:np.ndarray, ratings:pd.DataFrame, means, mu:float, epsilon:float, beta:float)->pd.DataFrame:
    L = degree - adjacency.to_numpy()
    identity = np.identity(degree.shape[0])
    X_star = np.linalg.inv(identity + mu * scipy.linalg.fractional_matrix_power((L + epsilon * identity), beta))@ ratings
    X_star = X_star.add(means.values, axis=0)
    X_star.index = adjacency.index
    return X_star


