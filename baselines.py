from ultis import extract_commit, filtering_commit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
import numpy as np


def extract_msg(commits):
    msgs = [" ".join(c["msg"].split(",")) for c in commits]
    return msgs


def extract_label(commits):
    labels = [1 if c["stable"] == "true" else 0 for c in commits]
    return labels


def get_items(items, indexes):
    return [items[i] for i in indexes]


def cross_validation(X, y, algorithm, folds):
    kf = KFold(n_splits=folds)
    kf.get_n_splits(X=X)
    for train_index, test_index in kf.split(X):
        # X_train, y_train = X[train_index], y[train_index]
        # X_test, y_test = X[test_index], y[test_index]
        X_train, y_train = get_items(items=X, indexes=train_index), get_items(items=y, indexes=train_index)
        X_test, y_test = get_items(items=X, indexes=test_index), get_items(items=y, indexes=test_index)

        # print len(X_train), len(y_train), len(X_test), len(y_test)
        vectorizer = CountVectorizer()
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)

        clf = LinearSVC(random_state=0)
        clf.fit(X=X_train, y=y_train)
        print accuracy_score(y_true=y_test, y_pred=clf.predict(X_test))
        print precision_score(y_true=y_test, y_pred=clf.predict(X_test))
        print recall_score(y_true=y_test, y_pred=clf.predict(X_test))
        print f1_score(y_true=y_test, y_pred=clf.predict(X_test))

        exit()

        vectorizer = CountVectorizer()
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)

        clf = LinearSVC(random_state=0)
        clf.fit(X=X_train, y=y_train)
        print accuracy_score(y_true=y_test, y_pred=clf.predict(X_test))


        print X_train.shape, y_train.shape
        print X_test.shape, y_test.shape
        # print len(train_index), len(test_index)
        # print len(train_index) + len(test_index)
        exit()


def baseline(train, label, algorithm, folds):
    # vectorizer = CountVectorizer()
    # X = vectorizer.fit_transform(train)
    cross_validation(X=train, y=label, algorithm=algorithm, folds=folds)
    # print X.shape
    # print type(X), type(np.array(label))
    # clf = LinearSVC(random_state=0)
    # clf.fit(X=X, y=label)
    # # print list(label)
    # # print clf.predict(X)
    # print accuracy_score(y_true=label, y_pred=clf.predict(X))
    # print precision_score(y_true=label, y_pred=clf.predict(X))
    # print recall_score(y_true=label, y_pred=clf.predict(X))
    # print f1_score(y_true=label, y_pred=clf.predict(X))
    exit()

if __name__ == "__main__":
    path_data = "./data/oct5/sample_eq100_line_oct5.out"
    # path_data = "./data/oct5/eq100_line_oct5.out"
    commits_ = extract_commit(path_file=path_data)
    nfile, nhunk, nline, nleng = 1, 8, 10, 120
    filter_commits = filtering_commit(commits=commits_, num_file=nfile, num_hunk=nhunk, num_loc=nline, size_line=nleng)
    msgs = extract_msg(commits=filter_commits)
    print len(msgs)
    labels = extract_label(commits=filter_commits)
    print len(labels)
    baseline(train=msgs, label=labels, algorithm="", folds=10)
