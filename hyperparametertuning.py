import pandas as pd
import numpy as np
from CF import predict_CF
from Tikhonov import k_nearest_symmetric, tikhonov, sobolev
from metrics import rmse, precision_at_k, recall_at_k, calculate_ndcg


def hyperparameter_tuning_cf(df_ratings:pd.DataFrame, df_test: pd.DataFrame, ns):
    df_simil = df_ratings.transpose().corr(method='pearson', min_periods=4)
    np.fill_diagonal(df_simil.values, 0)

    ndcgs = [[],[],[]]
    precisions = [[],[],[]]
    recalls = [[],[],[]]
    rmses = [[]]
    means = df_ratings.mean(axis=1)

    for n in ns:
        pred_cf = df_test.copy(deep=True)
        pred_cf["rating"] = 0
        pred_cf["rating"] = pred_cf.apply(lambda row: predict_CF(df_ratings, df_simil, n, row['user_id'], row["item_id"]), axis=1)
        df_pred_cf = pred_cf.pivot(index="user_id", columns="item_id", values="rating")

        rmses[0].append(rmse(df_pred_cf, df_test))
        precisions[0].append(precision_at_k(df_pred_cf, df_test, means.values, 10)[1])
        recalls[0].append(recall_at_k(df_pred_cf, df_test, means.values, 10)[1])
        ndcgs[0].append(calculate_ndcg(df_pred_cf, df_test, 10))

        precisions[1].append(precision_at_k(df_pred_cf, df_test, means.values, 5)[1])
        recalls[1].append(recall_at_k(df_pred_cf, df_test, means.values, 5)[1])
        ndcgs[1].append(calculate_ndcg(df_pred_cf, df_test, 5))

        precisions[2].append(precision_at_k(df_pred_cf, df_test, means.values, 20)[1])
        recalls[2].append(recall_at_k(df_pred_cf, df_test, means.values, 20)[1])
        ndcgs[2].append(calculate_ndcg(df_pred_cf, df_test, 20))

    return np.array(rmses), np.array(precisions), np.array(recalls), np.array(ndcgs)


def hyperparameter_tuning_tik(df_ratings:pd.DataFrame, means, df_test, ns, mus):
    missing_columns = missing_column(df_ratings, df_test, means)
    df_simil = df_ratings.transpose().corr(method='pearson', min_periods=4)
    np.fill_diagonal(df_simil.values, 0)
    df_ratings.fillna(0, inplace=True)
    ndcgs_tk = np.zeros((3,len(ns), len(mus)))
    precisions_tk = np.zeros((3,len(ns), len(mus)))
    recalls_tk = np.zeros((3,len(ns), len(mus)))
    rmses_tk = np.zeros((len(ns), len(mus)))

    for row in range(len(ns)):
        n = ns[row]
        sparse_df_simil, degree = k_nearest_symmetric(df_simil, n)
        for column in range(len(mus)):
            mu = mus[column]

            x_star_tk = tikhonov(sparse_df_simil, degree, df_ratings, means, mu)
            x_star_tk = x_star_tk.join(missing_columns)

            rmse_val_tk_10 = rmse(x_star_tk, df_test)
            precision_tk_10 = precision_at_k(x_star_tk, df_test, means.values, 10)
            recall_tk_10 = recall_at_k(x_star_tk, df_test, means.values, 10)
            ndcg_tk_10 = calculate_ndcg(x_star_tk, df_test, 10)

            precision_tk_5 = precision_at_k(x_star_tk, df_test, means.values, 5)
            recall_tk_5 = recall_at_k(x_star_tk, df_test, means.values, 5)
            ndcg_tk_5 = calculate_ndcg(x_star_tk, df_test, 5)

            precision_tk_20 = precision_at_k(x_star_tk, df_test, means.values, 20)
            recall_tk_20 = recall_at_k(x_star_tk, df_test, means.values, 20)
            ndcg_tk_20 = calculate_ndcg(x_star_tk, df_test, 20)

            rmses_tk[row][column] = rmse_val_tk_10
            precisions_tk[0][row][column] = precision_tk_10[1]
            ndcgs_tk[0][row][column] = ndcg_tk_10
            recalls_tk[0][row][column] = recall_tk_10[1]

            precisions_tk[1][row][column] = precision_tk_5[1]
            ndcgs_tk[1][row][column] = ndcg_tk_5
            recalls_tk[1][row][column] = recall_tk_5[1]

            precisions_tk[2][row][column] = precision_tk_20[1]
            ndcgs_tk[2][row][column] = ndcg_tk_20
            recalls_tk[2][row][column] = recall_tk_20[1]

    return rmses_tk, precisions_tk, recalls_tk, ndcgs_tk


def hyperparameter_tuning_sobolev(df_ratings:pd.DataFrame, means, df_test, ns, mus, beta, epsilon):
    missing_columns = missing_column(df_ratings, df_test, means)
    df_simil = df_ratings.transpose().corr(method='pearson', min_periods=4)
    np.fill_diagonal(df_simil.values, 0)
    df_ratings.fillna(0, inplace=True)

    ndcgs_sol = np.zeros((3, len(ns), len(mus)))
    precisions_sol = np.zeros((3, len(ns), len(mus)))
    recalls_sol = np.zeros((3, len(ns), len(mus)))
    rmses_sol = np.zeros((len(ns), len(mus)))

    for row in range(len(ns)):
        n = ns[row]
        sparse_df_simil, degree = k_nearest_symmetric(df_simil, n)
        for column in range(len(mus)):
            mu = mus[column]

            x_star_sol = sobolev(sparse_df_simil, degree, df_ratings, means, mu, epsilon, beta)
            x_star_sol = x_star_sol.join(missing_columns)

            rmse_val_sol_10 = rmse(x_star_sol, df_test)
            precision_sol_10 = precision_at_k(x_star_sol, df_test, means.values, 10)
            recall_sol_10 = recall_at_k(x_star_sol, df_test, means.values, 10)
            ndcg_sol_10 = calculate_ndcg(x_star_sol, df_test, 10)

            precision_sol_5 = precision_at_k(x_star_sol, df_test, means.values, 5)
            recall_sol_5 = recall_at_k(x_star_sol, df_test, means.values, 5)
            ndcg_sol_5 = calculate_ndcg(x_star_sol, df_test, 5)

            precision_sol_20 = precision_at_k(x_star_sol, df_test, means.values, 20)
            recall_sol_20 = recall_at_k(x_star_sol, df_test, means.values, 20)
            ndcg_sol_20 = calculate_ndcg(x_star_sol, df_test, 20)

            rmses_sol[row][column] = rmse_val_sol_10
            precisions_sol[0][row][column] = precision_sol_10[1]
            ndcgs_sol[0][row][column] = ndcg_sol_10
            recalls_sol[0][row][column] = recall_sol_10[1]

            precisions_sol[1][row][column] = precision_sol_5[1]
            ndcgs_sol[1][row][column] = ndcg_sol_5
            recalls_sol[1][row][column] = recall_sol_5[1]

            precisions_sol[2][row][column] = precision_sol_20[1]
            ndcgs_sol[2][row][column] = ndcg_sol_20
            recalls_sol[2][row][column] = recall_sol_20[1]

    return rmses_sol, precisions_sol, recalls_sol, ndcgs_sol


def missing_column(df_ratings:pd.DataFrame, df_test:pd.DataFrame, means:pd.DataFrame) -> pd.DataFrame:
    df_test_matrix = df_test.pivot(index="user_id", columns="item_id", values="rating")
    missing_columns = df_test_matrix[df_test_matrix.columns.difference(df_ratings.columns)].copy()
    filled = np.zeros(shape=(df_ratings.shape[0], missing_columns.shape[1])) + means.values
    df_filled = pd.DataFrame(filled, index=df_ratings.index, columns=missing_columns.columns)
    return df_filled
