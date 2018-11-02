import os
import sys

split_path, goback_tokens = os.getcwd().split("/"), 2
path_working = "/".join(split_path[:len(split_path) - goback_tokens])
print path_working
sys.path.append(path_working)

from ultis import extract_commit_july, filtering_commit, write_file
from baselines import extract_msg, extract_label, extract_code, add_two_list, baseline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from baselines_statistical_test import auc_score
from sklearn.model_selection import KFold
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from baselines import get_items, avg_list


def cross_validation_ver2(id, X, y, algorithm, folds):
    kf = KFold(n_splits=folds, random_state=None)
    kf.get_n_splits(X=X)
    accuracy, precision, recall, f1 = list(), list(), list(), list()
    probs = list()
    for train_index, test_index in kf.split(X):
        X_train, y_train = get_items(items=X, indexes=train_index), get_items(items=y, indexes=train_index)
        X_test, y_test = get_items(items=X, indexes=test_index), get_items(items=y, indexes=test_index)
        id_train, id_test = get_items(items=id, indexes=train_index), get_items(items=id, indexes=test_index)

        vectorizer = CountVectorizer()
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)
        # X = vectorizer.transform(X)

        # eval_train, eval_labels = loading_data("./data/3_mar7/typeaddres.out")
        # eval_train = vectorizer.transform(eval_train)

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
        accuracy.append(accuracy_score(y_true=y_test, y_pred=clf.predict(X_test)))
        precision.append(precision_score(y_true=y_test, y_pred=clf.predict(X_test)))
        recall.append(recall_score(y_true=y_test, y_pred=clf.predict(X_test)))
        f1.append(f1_score(y_true=y_test, y_pred=clf.predict(X_test)))
        # print accuracy, precision, recall, f1

        # print X_test.shape
        # y_pred = clf.predict(X_test)
        # y_pred_proba = clf.predict_proba(X_test)[:, 1]
        # y_pred_log_proba = clf.predict_log_proba(X_test)
        # print clf.predict_proba(X_test).shape
        # print clf.predict_log_proba(X_test).shape
        # exit()
        # probs += clf.predict_proba(X_test)[:, 1]
        probs = np.concatenate((probs, clf.predict_proba(X_test)[:, 1]), axis=0)

        # accuracy.append(accuracy_score(y_true=eval_labels, y_pred=clf.predict(eval_train)))
        # precision.append(precision_score(y_true=eval_labels, y_pred=clf.predict(eval_train)))
        # recall.append(recall_score(y_true=eval_labels, y_pred=clf.predict(eval_train)))
        # f1.append(f1_score(y_true=eval_labels, y_pred=clf.predict(eval_train)))
        # break

    print accuracy, "Accuracy of %s: %f" % (algorithm, avg_list(accuracy))
    print precision, "Precision of %s: %f" % (algorithm, avg_list(precision))
    print recall, "Recall of %s: %f" % (algorithm, avg_list(recall))
    print f1, "F1 of %s: %f" % (algorithm, avg_list(f1))

    path_write = "./statistical_test_prob/%s.txt" % (algorithm)
    write_file(path_file=path_write, data=probs)
    print len(probs)


def baseline_ver2(id, train, label, algorithm, folds):
    cross_validation_ver2(id=id, X=train, y=label, algorithm=algorithm, folds=folds)


def baseline_ver3(id, train, label, algorithm):
    X_train, y_train = train, label
    X_test, y_test = train, label
    id_train, id_test = id, id

    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    # X = vectorizer.transform(X)

    # eval_train, eval_labels = loading_data("./data/3_mar7/typeaddres.out")
    # eval_train = vectorizer.transform(eval_train)

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
    accuracy = accuracy_score(y_true=y_test, y_pred=clf.predict(X_test))
    precision = precision_score(y_true=y_test, y_pred=clf.predict(X_test))
    recall = recall_score(y_true=y_test, y_pred=clf.predict(X_test))
    f1 = f1_score(y_true=y_test, y_pred=clf.predict(X_test))
    auc = auc_score(y_true=y_test, y_pred=clf.predict(X_test))

    print "Accuracy:", accuracy
    print "Precision:", precision
    print "Recall:", recall
    print "F1:", f1
    print "AUC:", auc

    probs = clf.predict_proba(X_test)[:, 1]
    path_write = "./statistical_test_ver2/%s.txt" % (algorithm)
    write_file(path_file=path_write, data=probs)


if __name__ == "__main__":
    # path_data = "./typediff_sorted.out"
    # commits_ = extract_commit_july(path_file=path_data)
    # nfile, nhunk, nline, nleng = 1, 8, 10, 120
    # filter_commits = filtering_commit(commits=commits_, num_file=nfile, num_hunk=nhunk, num_loc=nline, size_line=nleng)
    # print len(commits_), len(filter_commits)

    # path_data = "./satisfy_typediff_sorted.out"
    # commits_ = extract_commit_july(path_file=path_data)
    # filter_commits = commits_
    #
    # msgs = extract_msg(commits=filter_commits)
    # labels = extract_label(commits=filter_commits)
    # codes = extract_code(commits=filter_commits)
    # all_lines = add_two_list(list1=msgs, list2=codes)
    # baseline(train=all_lines, label=labels, algorithm="svm", folds=5)
    # baseline(train=all_lines, label=labels, algorithm="lr", folds=5)
    # baseline(train=all_lines, label=labels, algorithm="dt", folds=5)

    #######################################################################################
    #######################################################################################
    path_data = "./satisfy_typediff_sorted.out"
    commits_ = extract_commit_july(path_file=path_data)
    filter_commits = commits_
    print len(filter_commits), type(filter_commits)
    commits_id = [c["id"] for c in commits_]
    print len(commits_id)

    msgs = extract_msg(commits=filter_commits)
    labels = extract_label(commits=filter_commits)
    codes = extract_code(commits=filter_commits)
    all_lines = add_two_list(list1=msgs, list2=codes)

    # baseline_ver2(id=commits_id, train=all_lines, label=labels, algorithm="lr", folds=5)
    baseline_ver3(id=commits_id, train=all_lines, label=labels, algorithm="lr")