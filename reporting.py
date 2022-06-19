import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def reporting_cf(scores, ns):
    hyperparam_rmse = pd.DataFrame(scores[0], index=[10], columns=ns)
    hyperparam_precision = pd.DataFrame(scores[1], index=[10,5,20], columns=ns)
    hyperparam_recall = pd.DataFrame(scores[2], index=[10,5,20], columns=ns)
    hyperparam_ndcg = pd.DataFrame(scores[3], index=[10,5,20], columns=ns)
    hyperparam_rmse.to_csv(f"./results/rmse_vals{time.time()}.csv")
    hyperparam_precision.to_csv(f"./results/precision_vals{time.time()}.csv")
    hyperparam_recall.to_csv(f"./results/recall_vals{time.time()}.csv")
    hyperparam_ndcg.to_csv(f"./results/ndcg_vals{time.time()}.csv")



def reporting(scores, ns, mus, method):
    hyperparam_rmse = pd.DataFrame(scores[0], index=ns, columns=mus)
    hyperparam_precision = pd.DataFrame(scores[1][0], index=ns, columns=mus)
    hyperparam_recall = pd.DataFrame(scores[2][0], index=ns, columns=mus)
    hyperparam_ndcg = pd.DataFrame(scores[3][0], index=ns, columns=mus)
    hyperparam_rmse.to_csv(f"./results/rmse_vals{time.time()}.csv")
    hyperparam_precision.to_csv(f"./results/10_precision_vals{time.time()}.csv")
    hyperparam_recall.to_csv(f"./results/10_recall_vals{time.time()}.csv")
    hyperparam_ndcg.to_csv(f"./results/10_ndcg_vals{time.time()}.csv")

    hyperparam_precision_5 = pd.DataFrame(scores[1][1], index=ns, columns=mus)
    hyperparam_recall_5 = pd.DataFrame(scores[2][1], index=ns, columns=mus)
    hyperparam_ndcg_5 = pd.DataFrame(scores[3][1], index=ns, columns=mus)
    hyperparam_precision_5.to_csv(f"./results/5_precision_vals{time.time()}.csv")
    hyperparam_recall_5.to_csv(f"./results/5_recall_vals{time.time()}.csv")
    hyperparam_ndcg_5.to_csv(f"./results/5_ndcg_vals{time.time()}.csv")

    hyperparam_precision_20 = pd.DataFrame(scores[1][2], index=ns, columns=mus)
    hyperparam_recall_20 = pd.DataFrame(scores[2][2], index=ns, columns=mus)
    hyperparam_ndcg_20 = pd.DataFrame(scores[3][2], index=ns, columns=mus)
    hyperparam_precision_20.to_csv(f"./results/20_precision_vals{time.time()}.csv")
    hyperparam_recall_20.to_csv(f"./results/20_recall_vals{time.time()}.csv")
    hyperparam_ndcg_20.to_csv(f"./results/20_ndcg_vals{time.time()}.csv")

    # nss = [f"k = {n}" for n in ns]
    nss = [f"n = {n}" for n in ns]
    lineObjectsRmse = plt.plot(mus, scores[0][0].T)
    plt.title(f'{method} RMSE values for different mu')
    plt.xlabel('$\mu$')
    plt.ylabel('RMSE')
    plt.legend(iter(lineObjectsRmse), nss)
    plt.savefig(f"./results/rmse{time.time()}.png")
    # plt.show()

    lineObjectsPrecision = plt.plot(mus, scores[1][0].T)
    plt.title(f'{method} precision@10 values for different mu')
    plt.xlabel('$\mu$')
    plt.ylabel('precision')
    plt.legend(iter(lineObjectsPrecision), nss)
    plt.savefig(f"./results/10_precision{time.time()}.png")
    # plt.show()

    lineObjectsRecall = plt.plot(mus, scores[2][0].T)
    plt.title(f'{method} recall@10 values for different mu')
    plt.xlabel('$\mu$')
    plt.ylabel('recall')
    plt.legend(iter(lineObjectsRecall), nss)
    plt.savefig(f"./results/10_recall{time.time()}.png")
    # plt.show()

    lineObjectsNDCG = plt.plot(mus, scores[3][0].T)
    plt.title(f'{method} NDCG@10 values for different mu')
    plt.xlabel('$\mu$')
    plt.ylabel('ndcg')
    plt.legend(iter(lineObjectsNDCG), nss)
    plt.savefig(f"./results/10_ndcg{time.time()}.png")
    # plt.show()

    lineObjectsPrecision_5 = plt.plot(mus, scores[1][1].T)
    plt.title(f'{method} precision@5 values for different mu')
    plt.xlabel('$\mu$')
    plt.ylabel('precision')
    plt.legend(iter(lineObjectsPrecision_5), nss)
    plt.savefig(f"./results/5_precision{time.time()}.png")
    # plt.show()

    lineObjectsRecall_5 = plt.plot(mus, scores[2][1].T)
    plt.title(f'{method} recall@5 values for different mu')
    plt.xlabel('$\mu$')
    plt.ylabel('recall')
    plt.legend(iter(lineObjectsRecall_5), nss)
    plt.savefig(f"./results/5_recall{time.time()}.png")
    # plt.show()

    lineObjectsNDCG_5 = plt.plot(mus, scores[3][1].T)
    plt.title(f'{method} NDCG@5 values for different mu')
    plt.xlabel('$\mu$')
    plt.ylabel('ndcg')
    plt.legend(iter(lineObjectsNDCG_5), nss)
    plt.savefig(f"./results/5_ndcg{time.time()}.png")
    # plt.show()

    lineObjectsPrecision_20 = plt.plot(mus, scores[1][2].T)
    plt.title(f'{method} precision@20 values for different mu')
    plt.xlabel('$\mu$')
    plt.ylabel('precision')
    plt.legend(iter(lineObjectsPrecision_20), nss)
    plt.savefig(f"./results/20_precision{time.time()}.png")
    # plt.show()

    lineObjectsRecall_20 = plt.plot(mus, scores[2][2].T)
    plt.title(f'{method} recall@20 values for different mu')
    plt.xlabel('$\mu$')
    plt.ylabel('recall')
    plt.legend(iter(lineObjectsRecall_20), nss)
    plt.savefig(f"./results/20_recall{time.time()}.png")
    # plt.show()

    lineObjectsNDCG_20 = plt.plot(mus, scores[3][2].T)
    plt.title(f'{method} NDCG@20 values for different mu')
    plt.xlabel('$\mu$')
    plt.ylabel('ndcg')
    plt.legend(iter(lineObjectsNDCG_20), nss)
    plt.savefig(f"./results/20_ndcg{time.time()}.png")
    # plt.show()

def reporting_sol(scores, ns, mus, beta, epsilon, method):
    hyperparam_rmse = pd.DataFrame(scores[0], index=ns, columns=mus)
    hyperparam_precision = pd.DataFrame(scores[1][0], index=ns, columns=mus)
    hyperparam_recall = pd.DataFrame(scores[2][0], index=ns, columns=mus)
    hyperparam_ndcg = pd.DataFrame(scores[3][0], index=ns, columns=mus)
    hyperparam_rmse.to_csv(f"./results/10_rmse_vals_b{beta}_e{epsilon}_t{time.time()}.csv")
    hyperparam_precision.to_csv(f"./results/10_precision_vals_b{beta}_e{epsilon}_t{time.time()}.csv")
    hyperparam_recall.to_csv(f"./results/10_recall_vals_b{beta}_e{epsilon}_t{time.time()}.csv")
    hyperparam_ndcg.to_csv(f"./results/10_ndcg_vals_b{beta}_e{epsilon}_t{time.time()}.csv")

    hyperparam_precision_5 = pd.DataFrame(scores[1][1], index=ns, columns=mus)
    hyperparam_recall_5 = pd.DataFrame(scores[2][1], index=ns, columns=mus)
    hyperparam_ndcg_5 = pd.DataFrame(scores[3][1], index=ns, columns=mus)
    hyperparam_precision_5.to_csv(f"./results/5_precision_vals_b{beta}_e{epsilon}_t{time.time()}.csv")
    hyperparam_recall_5.to_csv(f"./results/5_recall_vals_b{beta}_e{epsilon}_t{time.time()}.csv")
    hyperparam_ndcg_5.to_csv(f"./results/5_ndcg_vals_b{beta}_e{epsilon}_t{time.time()}.csv")

    hyperparam_precision_20 = pd.DataFrame(scores[1][2], index=ns, columns=mus)
    hyperparam_recall_20 = pd.DataFrame(scores[2][2], index=ns, columns=mus)
    hyperparam_ndcg_20 = pd.DataFrame(scores[3][2], index=ns, columns=mus)
    hyperparam_precision_20.to_csv(f"./results/20_precision_vals_b{beta}_e{epsilon}_t{time.time()}.csv")
    hyperparam_recall_20.to_csv(f"./results/20_recall_vals_b{beta}_e{epsilon}_t{time.time()}.csv")
    hyperparam_ndcg_20.to_csv(f"./results/20_ndcg_vals_b{beta}_e{epsilon}_t{time.time()}.csv")


    # nss = [f"k = {n}" for n in ns]
    nss = [f"n = {n}" for n in ns]
    lineObjectsRmse = plt.plot(mus, scores[0].T)
    plt.title(f'{method} RMSE values for different mu')
    plt.xlabel('$\mu$')
    plt.ylabel('RMSE')
    plt.legend(iter(lineObjectsRmse), nss)
    plt.savefig(f"./results/10_rmse_b{beta}_e{epsilon}_t{time.time()}.png")
    # plt.show()

    lineObjectsPrecision = plt.plot(mus, scores[1][0].T)
    plt.title(f'{method} precision@10 values for different mu')
    plt.xlabel('$\mu$')
    plt.ylabel('precision')
    plt.legend(iter(lineObjectsPrecision), nss)
    plt.savefig(f"./results/10_precision_b{beta}_e{epsilon}_t{time.time()}.png")
    # plt.show()

    lineObjectsRecall = plt.plot(mus, scores[2][0].T)
    plt.title(f'{method} recall@10 values for different mu')
    plt.xlabel('$\mu$')
    plt.ylabel('recall')
    plt.legend(iter(lineObjectsRecall), nss)
    plt.savefig(f"./results/10_recall_b{beta}_e{epsilon}_t{time.time()}.png")
    # plt.show()

    lineObjectsNDCG = plt.plot(mus, scores[3][0].T)
    plt.title(f'{method} NDCG@10 values for different mu')
    plt.xlabel('$\mu$')
    plt.ylabel('ndcg')
    plt.legend(iter(lineObjectsNDCG), nss)
    plt.savefig(f"./results/10_ndcg_b{beta}_e{epsilon}_t{time.time()}.png")
    # plt.show()

    lineObjectsPrecision_5 = plt.plot(mus, scores[1][1].T)
    plt.title(f'{method} precision@5 values for different mu')
    plt.xlabel('$\mu$')
    plt.ylabel('precision')
    plt.legend(iter(lineObjectsPrecision_5), nss)
    plt.savefig(f"./results/5_precision_b{beta}_e{epsilon}_t{time.time()}.png")
    # plt.show()

    lineObjectsRecall_5 = plt.plot(mus, scores[2][1].T)
    plt.title(f'{method} recall@5 values for different mu')
    plt.xlabel('$\mu$')
    plt.ylabel('recall')
    plt.legend(iter(lineObjectsRecall_5), nss)
    plt.savefig(f"./results/5_recall_b{beta}_e{epsilon}_t{time.time()}.png")
    # plt.show()

    lineObjectsNDCG_5 = plt.plot(mus, scores[3][1].T)
    plt.title(f'{method} NDCG@5 values for different mu')
    plt.xlabel('$\mu$')
    plt.ylabel('ndcg')
    plt.legend(iter(lineObjectsNDCG_5), nss)
    plt.savefig(f"./results/5_ndcg_b{beta}_e{epsilon}_t{time.time()}.png")
    # plt.show()

    lineObjectsPrecision_20 = plt.plot(mus, scores[1][2].T)
    plt.title(f'{method} precision@20 values for different mu')
    plt.xlabel('$\mu$')
    plt.ylabel('precision')
    plt.legend(iter(lineObjectsPrecision_20), nss)
    plt.savefig(f"./results/5_precision_20{beta}_e{epsilon}_t{time.time()}.png")
    # plt.show()

    lineObjectsRecall_20 = plt.plot(mus, scores[2][2].T)
    plt.title(f'{method} recall@20 values for different mu')
    plt.xlabel('$\mu$')
    plt.ylabel('recall')
    plt.legend(iter(lineObjectsRecall_20), nss)
    plt.savefig(f"./results/20_recall_b{beta}_e{epsilon}_t{time.time()}.png")
    # plt.show()

    lineObjectsNDCG_20 = plt.plot(mus, scores[3][2].T)
    plt.title(f'{method} NDCG@20 values for different mu')
    plt.xlabel('$\mu$')
    plt.ylabel('ndcg')
    plt.legend(iter(lineObjectsNDCG_20), nss)
    plt.savefig(f"./results/20_ndcg_b{beta}_e{epsilon}_t{time.time()}.png")
    # plt.show()

def plot_csv(path: str, axis_name: str):
    df = pd.read_csv(path, index_col=0)
    # 8 colours
    colours = ["#EC929F", "#D39ABF", "#A3A9D2", "#68B6CD", "#41BDB0", "#5CBE84", "#8EB857", "#C2AB3D"]
    # 96 colours
    # colours = ["#E68F86", "#E78E89", "#E78E8C", "#E78E8E", "#E68E91", "#E68E94", "#E58E97", "#E58E9A", "#E48E9D", "#E28EA0", "#E18FA3", "#DF8FA6", "#DE90A8", "#DC90AB", "#DA91AE", "#D792B1", "#D593B4", "#D294B6", "#CF95B9", "#CC96BC", "#C997BE", "#C598C0", "#C299C2", "#BE9BC5", "#BA9CC6", "#B59DC8", "#B19ECA", "#ADA0CB", "#A8A1CD", "#A3A2CE", "#9EA4CF", "#99A5D0", "#94A6D0", "#8EA8D1", "#89A9D1", "#83AAD1", "#7DABD1", "#77ACD0", "#71AED0", "#6BAFCF", "#65B0CE", "#5FB1CD", "#59B2CB", "#53B3CA", "#4DB4C8", "#47B4C6", "#41B5C4", "#3BB6C1", "#35B7BF", "#2FB7BC", "#2AB8B9", "#25B8B6", "#22B9B2", "#1FB9AF", "#1DBAAB", "#1DBAA8", "#1FBAA4", "#21BBA0", "#25BB9C", "#29BB98", "#2EBB94", "#33BB8F", "#38BB8B", "#3DBB86", "#42BB82", "#48BB7D", "#4DBA79", "#52BA74", "#57BA70", "#5DB96B", "#62B967", "#67B862", "#6CB85E", "#71B759", "#76B755", "#7BB651", "#81B54D", "#86B449", "#8BB345", "#90B241", "#95B13D", "#9AB03A", "#9FAF36", "#A4AE33", "#A9AC30", "#AEAB2E", "#B3A92B", "#B8A829", "#BDA628", "#C1A427", "#C6A326", "#CBA126", "#D09F27", "#D49D28", "#D99B29", "#DD992B", "#E1962D", "#E69430"]
    plt.rcParams.update({'font.size': 14})
    lineobjects = plt.plot(np.arange(0.0, 1.51, 0.01), df.values.T,linewidth=2.0)
    for i,j in enumerate(lineobjects):
        j.set_color(colours[i])
    # To place legend outside of plot
    plt.legend(iter(lineobjects), df.index, loc=(1.04, 0))
    plt.xlabel('$\mu$')
    plt.ylabel(axis_name)
    plt.grid()
    plt.savefig(f"./results/{axis_name}.pdf", bbox_inches='tight')
    plt.show()
