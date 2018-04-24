from ultis import load_file
from baselines import extract_commit, filtering_commit, extract_label, get_items
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from baselines_statistical_test import auc_score


def kfold_results(y_true, y_pred, nfolds):
    kf = KFold(n_splits=nfolds, random_state=0)
    accuracy, precision, recall, f1, auc = list(), list(), list(), list(), list()
    for train_index, test_index in kf.split(y_true):
        y_true_test_index, y_pred_test_index = get_items(items=y_true, indexes=test_index), \
                                               get_items(items=y_pred, indexes=test_index)
        accuracy.append(accuracy_score(y_true=y_true_test_index, y_pred=y_pred_test_index))
        precision.append(precision_score(y_true=y_true_test_index, y_pred=y_pred_test_index))
        recall.append(recall_score(y_true=y_true_test_index, y_pred=y_pred_test_index))
        f1.append(f1_score(y_true=y_true_test_index, y_pred=y_pred_test_index))
        auc.append(auc_score(y_true=y_true_test_index, y_pred=y_pred_test_index))

    algorithm = ""
    print "Accuracy and std of %s: %f %f" % (algorithm, np.mean(np.array(accuracy)), np.std(np.array(accuracy)))
    print "Precision of %s: %f %f" % (algorithm, np.mean(np.array(precision)), np.std(np.array(precision)))
    print "Recall of %s: %f %f" % (algorithm, np.mean(np.array(recall)), np.std(np.array(recall)))
    print "F1 of %s: %f %f" % (algorithm, np.mean(np.array(f1)), np.std(np.array(f1)))
    print "AUC of %s: %f %f" % (algorithm, np.mean(np.array(auc)), np.std(np.array(auc)))


if __name__ == "__main__":
    path_model = "./statistical_test_ver2/3_mar7/fold_0_1521433495_model-48550.txt"
    results_model = map(float, load_file(path_file=path_model))
    print len(results_model)

    path_data = "./data/3_mar7/typediff.out"
    commits_ = extract_commit(path_file=path_data)
    nfile, nhunk, nline, nleng = 1, 8, 10, 120
    filter_commits = filtering_commit(commits=commits_, num_file=nfile, num_hunk=nhunk, num_loc=nline, size_line=nleng)
    labels = extract_label(commits=filter_commits)
    print len(filter_commits)
    kfold_results(y_true=labels, y_pred=results_model, nfolds=5)
