from ultis import load_file
import numpy as np
from baselines_statistical_test import auc_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from train_PatchNet import split_train_test
from baselines import get_items

if __name__ == "__main__":
    path_true = "./statistical_test_prob/true_label.txt"
    y_true = load_file(path_file=path_true)
    y_true = np.array([int(y) for y in y_true])
    folds, random_state = 5, None

    path_pred = "./statistical_test_prob/lstm_cnn_all.txt"
    # path_pred = "./statistical_test_prob/lstm_cnn_msg.txt"
    # path_pred = "./statistical_test_prob/lstm_cnn_code.txt"
    # path_pred = "./statistical_test_prob/cnn_all.txt"
    # path_pred = "./statistical_test_prob/cnn_msg.txt"
    # path_pred = "./statistical_test_prob/cnn_code.txt"
    # path_pred = "./statistical_test_prob/msg_model-21550.txt"
    # path_pred = "./statistical_test_prob/code_model-20687.txt"
    # path_pred = "./statistical_test_prob/all_model-19825.txt"
    # path_pred = "./statistical_test_prob/lr.txt"
    # path_pred = "./statistical_test_prob/lstm_cnn_all_fold_0.txt"
    # path_pred = "./statistical_test_prob/lstm_cnn_all_check_fold_0.txt"
    path_pred = "./statistical_test_prob/lstm_cnn_all_checking.txt"
    y_pred = load_file(path_file=path_pred)
    y_pred = np.array([float(y) for y in y_pred])
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0

    split_data = split_train_test(data=y_true, folds=folds, random_state=None)

    for i in xrange(len(split_data)):
        train_index, test_index = split_data[i]["train"], split_data[i]["test"]
        y_true_, y_pred_ = get_items(items=y_true, indexes=test_index), get_items(items=y_pred, indexes=test_index)
        acc = accuracy_score(y_true=y_true_, y_pred=y_pred_)
        prc = precision_score(y_true=y_true_, y_pred=y_pred_)
        rc = recall_score(y_true=y_true_, y_pred=y_pred_)
        f1 = f1_score(y_true=y_true_, y_pred=y_pred_)
        auc = auc_score(y_true=y_true_, y_pred=y_pred_)
        print acc, prc, rc, f1, auc

