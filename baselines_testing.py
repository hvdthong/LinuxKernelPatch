from ultis import extract_commit, filtering_commit
from baselines import extract_msg, extract_label, extract_code, add_two_list
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def baseline_testing(X_train, y_train, X_test, y_test, algorithm):
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    print X_train.shape, X_test.shape

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
    print accuracy_score(y_true=y_test, y_pred=clf.predict(X_test))


if __name__ == "__main__":
    nfile, nhunk, nline, nleng = 1, 8, 10, 120

    path_data = "./data/3_mar7/typediff.out"
    commits_train = extract_commit(path_file=path_data)
    filter_commits_train = filtering_commit(commits=commits_train, num_file=nfile,
                                            num_hunk=nhunk, num_loc=nline,
                                            size_line=nleng)
    msgs_train = extract_msg(commits=filter_commits_train)
    labels_train = extract_label(commits=filter_commits_train)
    codes_train = extract_code(commits=filter_commits_train)
    all_lines_train = add_two_list(list1=msgs_train, list2=codes_train)

    # path_test = "./data/test_data/sasha_translated.out"
    path_test = "./data/test_data/markus_translated.out"
    commits_test = extract_commit(path_file=path_test)
    filter_commits_test = filtering_commit(commits=commits_test,
                                           num_file=nfile, num_hunk=nhunk,
                                           num_loc=nline, size_line=nleng)
    msgs_test = extract_msg(commits=filter_commits_test)
    labels_test = extract_label(commits=filter_commits_test)
    codes_test = extract_code(commits=filter_commits_test)
    all_lines_test = add_two_list(list1=msgs_test, list2=codes_test)

    print path_test
    baseline_testing(X_train=all_lines_train, y_train=labels_train,
                     X_test=all_lines_test, y_test=labels_test,
                     algorithm="svm")
