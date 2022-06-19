import time
import sys
import numpy as np
import pandas as pd
from hyperparametertuning import hyperparameter_tuning_tik, hyperparameter_tuning_cf, hyperparameter_tuning_sobolev
from preprocessing import mean_centering, train_test_data_from_path
from reporting import reporting, reporting_cf, reporting_sol



def main(method, fold=10, beta=1.0, epsilon=0.0):
    print(method, fold, beta, epsilon)
    return
    average_scores_rmse = None
    average_scores_precision = None
    average_scores_recall = None
    average_scores_ndcg = None
    method_name = "CF"
    if method == 1:
        method_name = "Tikhonov"
    elif method == 1:
        method = "Sobolev"

    ns = np.arange(5,41,5)
    mus = np.arange(0.0, 1.51, 0.01)
    for i in range(fold):
        print(f"fold = {i}")
        if method == 0:
            scores = cf_fold(ns)
        elif method == 1:
            scores = tikhonov_fold(ns, mus)
        elif method == 2:
            scores = sobolev_fold(ns, mus, beta, epsilon)
        else:
            print("Please choose 0 for collaborative filtering, 1 for Tikhonov or 2 for Sobolev")
            return -1
        if average_scores_rmse is None:
            average_scores_rmse = scores[0]
        else:
            average_scores_rmse = average_scores_rmse + scores[0]
        if average_scores_precision is None:
            average_scores_precision = scores[1]
        else:
            average_scores_precision = average_scores_precision + scores[1]
        if average_scores_recall is None:
            average_scores_recall = scores[2]
        else:
            average_scores_recall = average_scores_recall + scores[2]
        if average_scores_ndcg is None:
            average_scores_ndcg = scores[3]
        else:
            average_scores_ndcg = average_scores_ndcg + scores[3]
    scores = (average_scores_rmse/fold, average_scores_precision/fold, average_scores_recall/fold, average_scores_ndcg/fold)
    if method == 0:
        reporting_cf(scores, ns)
    elif method == 1:
        reporting(scores, ns, mus, method_name)
    elif method == 2:
        reporting_sol(scores, ns, mus, beta, epsilon, method_name)


def cf_fold(ns):
    start = time.time()
    print("start")
    df_train_raw, df_test = train_test_data_from_path("./ml-100k/u.data")
    print("tuning")
    scores = hyperparameter_tuning_cf(df_train_raw, df_test, ns)
    print("end tuning")
    print(scores)
    reporting_cf(scores, ns)
    end = time.time()
    print(end - start)
    return scores


def tikhonov_fold(ns, mus):
    start = time.time()
    print("start")
    df_train_raw, df_test = train_test_data_from_path("./ml-100k/u.data")

    df_train, means = mean_centering(df_train_raw)
    means = pd.DataFrame(means, index=df_train.index)

    print("tuning")
    scores = hyperparameter_tuning_tik(df_train, means, df_test, ns, mus)
    print("end tuning")
    reporting(scores, ns, mus, "Tikhonov")
    end = time.time()
    print(end - start)
    return scores


def sobolev_fold(ns, mus, beta, epsilon):
    start = time.time()
    print("start")
    df_train_raw, df_test = train_test_data_from_path("./ml-100k/u.data")

    df_train, means = mean_centering(df_train_raw)
    means = pd.DataFrame(means, index=df_train.index)

    print("tuning")
    scores = hyperparameter_tuning_sobolev(df_train, means, df_test, ns, mus, beta, epsilon)
    print("end tuning")
    reporting_sol(scores, ns, mus, beta, epsilon, "Sobolev")
    end = time.time()
    print(end - start)
    return scores

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Please choose 0 for collaborative filtering, 1 for Tikhonov or 2 for Sobolev")
    if len(sys.argv) == 2:
        main(int(sys.argv[1]))
    if len(sys.argv) == 3:
        main(int(sys.argv[1]), int(sys.argv[2]))
    if len(sys.argv) == 5:
        main(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
