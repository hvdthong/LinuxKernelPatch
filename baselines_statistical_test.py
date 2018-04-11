from ultis import extract_commit, write_file
from baselines import extract_msg, extract_label, extract_code, add_two_list
from sklearn.model_selection import KFold
from baselines import get_items
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from ultis import filtering_commit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn import metrics
from baselines import avg_list
import numpy as np


def auc_score(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true=y_true, y_score=y_pred, pos_label=1)
    return metrics.auc(fpr, tpr)


def make_dictionary(y_pred, y_index):
    pred_dict = dict()
    for i, j in zip(y_pred, y_index):
        pred_dict[j] = i
    return pred_dict


def sorted_dict(dict):
    data = list()
    for key in sorted(dict):
        data.append(dict[key])
    return data


def get_predict(name, X, y, algorithm, folds):
    kf = KFold(n_splits=folds, random_state=0)
    kf.get_n_splits(X=X)
    auc, accuracy, precision, recall, f1 = list(), list(), list(), list(), list()
    pred_dict = dict()
    for train_index, test_index in kf.split(X):
        X_train, y_train = get_items(items=X, indexes=train_index), get_items(items=y, indexes=train_index)
        X_test, y_test = get_items(items=X, indexes=test_index), get_items(items=y, indexes=test_index)

        vectorizer = CountVectorizer()
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)

        if algorithm == "svm":
            clf = LinearSVC(random_state=0)
        elif algorithm == "lr":
            clf = LogisticRegression()
        elif algorithm == "dt":
            clf = DecisionTreeClassifier()
        else:
            print "Wrong algorithm name -- please retype again"
            exit()

        clf.fit(X=X_train, y=y_train)
        y_pred = clf.predict(X_test)
        pred_dict.update(make_dictionary(y_pred=y_pred, y_index=test_index))
        # path_file = "./statistical_test/" + name + "_" + algorithm + ".txt"
        # path_file = "./statistical_test/3_mar7/" + name + "_" + algorithm + ".txt"
        # path_file = "./statistical_test_ver2/3_mar7" + name + "_" + algorithm + ".txt"
        # write_file(path_file, y_pred)
        accuracy.append(accuracy_score(y_true=y_test, y_pred=y_pred))
        precision.append(precision_score(y_true=y_test, y_pred=y_pred))
        recall.append(recall_score(y_true=y_test, y_pred=y_pred))
        f1.append(f1_score(y_true=y_test, y_pred=y_pred))
        auc.append(auc_score(y_true=y_test, y_pred=y_pred))

    path_file = "./statistical_test_ver2/3_mar7/" + name + "_" + algorithm + ".txt"
    write_file(path_file=path_file, data=sorted_dict(dict=pred_dict))
    print "Accuracy and std of %s: %f %f" % (algorithm, np.mean(np.array(accuracy)), np.std(np.array(accuracy)))
    print "Precision of %s: %f %f" % (algorithm, np.mean(np.array(precision)), np.std(np.array(precision)))
    print "Recall of %s: %f %f" % (algorithm, np.mean(np.array(recall)), np.std(np.array(recall)))
    print "F1 of %s: %f %f" % (algorithm, np.mean(np.array(f1)), np.std(np.array(f1)))
    print "AUC of %s: %f %f" % (algorithm, np.mean(np.array(auc)), np.std(np.array(auc)))


def predict_test_data(name, train, label, algorithm, folds):
    get_predict(name=name, X=train, y=label, algorithm=algorithm, folds=folds)


def print_label_data(path, name, commits):
    labels_data = [c["stable"] for c in commits]
    labels_data = [1 if "true" == l else 0 for l in labels_data]
    write_file(path_file=path + "/" + name, data=labels_data)
    return None


if __name__ == "__main__":
    # path_data = "./data/1_oct5/eq100_line_oct5.out"
    # path_data = "./data/1_oct5/sample_eq100_line_oct5.out"
    path_data = "./data/3_mar7/typediff.out"
    commits_ = extract_commit(path_file=path_data)
    nfile, nhunk, nline, nleng = 1, 8, 10, 120
    filter_commits = filtering_commit(commits=commits_, num_file=nfile, num_hunk=nhunk, num_loc=nline, size_line=nleng)
    print len(filter_commits)
    # filter_commits = commits_
    msgs = extract_msg(commits=filter_commits)
    labels = extract_label(commits=filter_commits)
    predict_test_data(name="msg", train=msgs, label=labels, algorithm="svm", folds=5)
    predict_test_data(name="msg", train=msgs, label=labels, algorithm="lr", folds=5)
    predict_test_data(name="msg", train=msgs, label=labels, algorithm="dt", folds=5)

    ############################################################################
    msgs = extract_msg(commits=filter_commits)
    labels = extract_label(commits=filter_commits)
    codes = extract_code(commits=filter_commits)
    all_lines = add_two_list(list1=msgs, list2=codes)
    predict_test_data(name="msg_code", train=all_lines, label=labels, algorithm="svm", folds=5)
    predict_test_data(name="msg_code", train=all_lines, label=labels, algorithm="lr", folds=5)
    predict_test_data(name="msg_code", train=all_lines, label=labels, algorithm="dt", folds=5)

    ############################################################################
    msgs = extract_msg(commits=filter_commits)
    labels = extract_label(commits=filter_commits)
    codes = extract_code(commits=filter_commits)
    all_lines = add_two_list(list1=msgs, list2=codes)
    predict_test_data(name="code", train=codes, label=labels, algorithm="svm", folds=5)
    predict_test_data(name="code", train=codes, label=labels, algorithm="lr", folds=5)
    predict_test_data(name="code", train=codes, label=labels, algorithm="dt", folds=5)
