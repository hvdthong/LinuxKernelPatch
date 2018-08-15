from os import listdir
from os.path import isfile, join
from ultis import load_file
import numpy as np
from baselines_statistical_test import auc_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def best_performance(pred, true):
    f1 = f1_score(y_true=pred, y_pred=true)
    return f1


if __name__ == "__main__":
    path_file = "./patchNet_mergeResults/"
    onlyfiles = [f for f in listdir(path_file) if isfile(join(path_file, f))]
    # for f in onlyfiles:
    #     print f

    path_true = "./statistical_test_prob/true_label.txt"
    y_true = load_file(path_file=path_true)
    y_true = np.array([int(y) for y in y_true])
    folds, random_state = 5, None
    scores, model_name = list(), "code"
    files_model_name = list()
    for f in onlyfiles:
        if model_name == "msg" or model_name == "code" or model_name == "all":
            if model_name in f:
                y_pred = load_file(path_file + f)
                y_pred = np.array([float(y) for y in y_pred])
                y_pred[y_pred > 0.5] = 1
                y_pred[y_pred <= 0.5] = 0
                scores.append(best_performance(pred=y_pred, true=y_true))
                files_model_name.append(f)
        else:
            print "You need to type correct model_name"
            exit()
    max_file = files_model_name[scores.index(max(scores))]
    print max_file
