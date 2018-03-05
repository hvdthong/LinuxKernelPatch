from ultis import extract_commit, filtering_commit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def extract_msg(commits):
    msgs = [" ".join(c["msg"].split(",")) for c in commits]
    return msgs


def extract_label(commits):
    labels = [1 if c["stable"] == "true" else 0 for c in commits]
    return labels


def extract_line_code(dict_code):
    lines = list()
    for k in dict_code.keys():
        for l in dict_code[k]:
            lines += l.split(":")[1].split(",")
    return lines


def extract_code(commits):
    codes = list()
    for c in commits:
        line = list()
        for t in c["code"]:
            added_line, removed_line = extract_line_code(t["added"]), extract_line_code(t["removed"])
            line += added_line + removed_line
        codes.append(" ".join(line))
    return codes


def add_two_list(list1, list2):
    lines = list()
    if len(list1) != len(list2):
        print "Your lists don't have a same length"
        exit()
    else:
        for a, b in zip(list1, list2):
            lines.append(a + " " + b)
    return lines


def get_items(items, indexes):
    return [items[i] for i in indexes]


def avg_list(numbers):
    return float(sum(numbers)) / len(numbers)


def cross_validation(X, y, algorithm, folds):
    kf = KFold(n_splits=folds, random_state=0)
    kf.get_n_splits(X=X)
    accuracy, precision, recall, f1 = list(), list(), list(), list()
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
        accuracy.append(accuracy_score(y_true=y_test, y_pred=clf.predict(X_test)))
        precision.append(precision_score(y_true=y_test, y_pred=clf.predict(X_test)))
        recall.append(recall_score(y_true=y_test, y_pred=clf.predict(X_test)))
        f1.append(f1_score(y_true=y_test, y_pred=clf.predict(X_test)))
        break

    print "Accuracy of %s: %f" % (algorithm, avg_list(accuracy))
    print "Precision of %s: %f" % (algorithm, avg_list(precision))
    print "Recall of %s: %f" % (algorithm, avg_list(recall))
    print "F1 of %s: %f" % (algorithm, avg_list(f1))


def baseline(train, label, algorithm, folds):
    cross_validation(X=train, y=label, algorithm=algorithm, folds=folds)


if __name__ == "__main__":
    # path_data = "./data/1_oct5/sample_eq100_line_oct5.out"
    path_data = "./data/1_oct5/eq100_line_oct5.out"
    # path_data = "./data/2_feb9/newres.out"
    commits_ = extract_commit(path_file=path_data)
    # nfile, nhunk, nline, nleng = 1, 8, 10, 120
    # filter_commits = filtering_commit(commits=commits_, num_file=nfile, num_hunk=nhunk, num_loc=nline, size_line=nleng)
    filter_commits = commits_
    ############################################################################
    msgs = extract_msg(commits=filter_commits)
    labels = extract_label(commits=filter_commits)
    baseline(train=msgs, label=labels, algorithm="svm", folds=5)
    baseline(train=msgs, label=labels, algorithm="lr", folds=5)
    baseline(train=msgs, label=labels, algorithm="dt", folds=5)
    ############################################################################
    msgs = extract_msg(commits=filter_commits)
    labels = extract_label(commits=filter_commits)
    codes = extract_code(commits=filter_commits)
    all_lines = add_two_list(list1=msgs, list2=codes)
    baseline(train=all_lines, label=labels, algorithm="svm", folds=5)
    baseline(train=all_lines, label=labels, algorithm="lr", folds=5)
    baseline(train=all_lines, label=labels, algorithm="dt", folds=5)
