from ultis import extract_commit, write_file
from baselines import extract_msg, extract_label, extract_code, add_two_list
from sklearn.model_selection import KFold
from baselines import get_items
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from ultis import filtering_commit


def get_predict(name, X, y, algorithm, folds):
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
        y_pred = clf.predict(X_test)
        path_file = "./statistical_test/" + name + "_" + algorithm + ".txt"
        write_file(path_file, y_pred)
        break


def predict_test_data(name, train, label, algorithm, folds):
    get_predict(name=name, X=train, y=label, algorithm=algorithm, folds=folds)


if __name__ == "__main__":
    path_data = "./data/1_oct5/eq100_line_oct5.out"
    # path_data = "./data/1_oct5/sample_eq100_line_oct5.out"
    commits_ = extract_commit(path_file=path_data)
    nfile, nhunk, nline, nleng = 1, 8, 10, 120
    filter_commits = filtering_commit(commits=commits_, num_file=nfile, num_hunk=nhunk, num_loc=nline, size_line=nleng)
    # filter_commits = commits_
    msgs = extract_msg(commits=filter_commits)
    labels = extract_label(commits=filter_commits)
    predict_test_data(name="msg", train=msgs, label=labels, algorithm="svm", folds=10)
    predict_test_data(name="msg", train=msgs, label=labels, algorithm="lr", folds=10)
    predict_test_data(name="msg", train=msgs, label=labels, algorithm="dt", folds=10)

    ############################################################################
    msgs = extract_msg(commits=filter_commits)
    labels = extract_label(commits=filter_commits)
    codes = extract_code(commits=filter_commits)
    all_lines = add_two_list(list1=msgs, list2=codes)
    predict_test_data(name="msg_code", train=all_lines, label=labels, algorithm="svm", folds=10)
    predict_test_data(name="msg_code", train=all_lines, label=labels, algorithm="lr", folds=10)
    predict_test_data(name="msg_code", train=all_lines, label=labels, algorithm="dt", folds=10)

    ############################################################################
    msgs = extract_msg(commits=filter_commits)
    labels = extract_label(commits=filter_commits)
    codes = extract_code(commits=filter_commits)
    all_lines = add_two_list(list1=msgs, list2=codes)
    predict_test_data(name="code", train=codes, label=labels, algorithm="svm", folds=10)
    predict_test_data(name="code", train=codes, label=labels, algorithm="lr", folds=10)
    predict_test_data(name="code", train=codes, label=labels, algorithm="dt", folds=10)

