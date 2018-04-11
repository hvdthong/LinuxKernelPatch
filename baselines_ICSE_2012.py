from baselines import extract_commit, filtering_commit
from ultis import load_file, write_file
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from random import shuffle
from sklearn import preprocessing
from ultis import get_commits
from baselines import extract_msg, extract_code, add_two_list
from sklearn.feature_extraction.text import CountVectorizer
from baselines_statistical_test import make_dictionary, auc_score, sorted_dict


def load_ftr(data):
    ids, ftrs = list(), list()
    for d in data:
        split_ = d.split(",")
        id_, ftr_ = split_[0], map(int, split_[1:])
        ids.append(id_)
        ftrs.append(np.array(ftr_))
    return ids, np.array(ftrs)


def get_data(ids_overlap_, ids_ftr_, data_ftr_, ids_data_, labes_data_):
    data = list()
    for i in ids_overlap_:
        index_ftr = ids_ftr_.index(i)
        label = labes_data_[ids_data_.index(i)]
        data.append(data_ftr_[index_ftr].strip() + "," + label)
    return data


def balance_data_ICSE(path):
    data = load_file(path_file=path)
    new_data, cnt = list(), 0
    for d in data:
        if "true" in d and cnt <= 11165:
            new_data.append(d.strip())
            cnt += 1
        elif "false" in d:
            new_data.append(d.strip())
    shuffle(new_data)
    write_file(path_file="./data/3_mar7/new_features_ver1.txt", data=new_data)
    exit()


def load_data_ICSE(path):
    data = load_file(path_file=path)
    ids, ftrs, labels = list(), list(), list()
    for d in data:
        split_ = d.split(",")
        id_, ftr_ = split_[0], map(int, split_[1:len(split_) - 1])
        label_ = split_[len(split_) - 1]
        ids.append(id_)
        ftrs.append(np.array(ftr_))
        labels.append(label_)
    labels = [1 if v.strip() == "true" else 0 for v in labels]
    return ids, np.array(ftrs), np.array(labels)


def create_features_ICSE(commits, ids, type):
    new_commits = list()
    for id_ in ids:
        for c in commits:
            if c["id"] == id_:
                new_commits.append(c)
                break
    vectorizer = CountVectorizer()
    if type == "msg":
        msgs = extract_msg(commits=new_commits)
        X = vectorizer.fit_transform(msgs)
    elif type == "code":
        codes = extract_code(commits=new_commits)
        X = vectorizer.fit_transform(codes)
    elif type == "all":
        msgs = extract_msg(commits=new_commits)
        codes = extract_code(commits=new_commits)
        all_lines = add_two_list(list1=msgs, list2=codes)
        X = vectorizer.fit_transform(all_lines)
    else:
        print "Your type is uncorrect"
        exit()
    return X.toarray()


def get_predict_ICSE(name, X, y, algorithm, folds):
    kf = KFold(n_splits=folds, random_state=0)
    kf.get_n_splits(X=X)
    accuracy, precision, recall, f1, auc = list(), list(), list(), list(), list()
    X = preprocessing.normalize(X)
    pred_dict = dict()
    for train_index, test_index in kf.split(X):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

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
        accuracy.append(accuracy_score(y_true=y_test, y_pred=y_pred))
        precision.append(precision_score(y_true=y_test, y_pred=y_pred))
        recall.append(recall_score(y_true=y_test, y_pred=y_pred))
        f1.append(f1_score(y_true=y_test, y_pred=y_pred))
        auc.append(auc_score(y_true=y_test, y_pred=y_pred))

        # y_pred = clf.predict(X)
        # path_file = "./data/3_mar7/" + "new_features_ver1_pred.txt"
        # write_file(path_file, y_pred)
        # break
    # print "Accuracy of %s: %f" % (algorithm, avg_list(accuracy))
    # print "Precision of %s: %f" % (algorithm, avg_list(precision))
    # print "Recall of %s: %f" % (algorithm, avg_list(recall))
    # print "F1 of %s: %f" % (algorithm, avg_list(f1))


    path_file = "./data/3_mar7/" + "new_features_ver2_pred.txt"
    write_file(path_file=path_file, data=sorted_dict(dict=pred_dict))
    print "Accuracy and std of %s: %f %f" % (algorithm, np.mean(np.array(accuracy)), np.std(np.array(accuracy)))
    print "Precision of %s: %f %f" % (algorithm, np.mean(np.array(precision)), np.std(np.array(precision)))
    print "Recall of %s: %f %f" % (algorithm, np.mean(np.array(recall)), np.std(np.array(recall)))
    print "F1 of %s: %f %f" % (algorithm, np.mean(np.array(f1)), np.std(np.array(f1)))
    print "AUC of %s: %f %f" % (algorithm, np.mean(np.array(auc)), np.std(np.array(auc)))


if __name__ == "__main__":
    # # path_data = "./data/3_mar7/typeaddres.out"
    # path_data = "./data/3_mar7/typediff.out"
    # commits_ = extract_commit(path_file=path_data)
    # nfile, nhunk, nline, nleng = 1, 8, 10, 120
    # commits_ = filtering_commit(commits=commits_, num_file=nfile, num_hunk=nhunk, num_loc=nline, size_line=nleng)
    # ids_data = [c["id"] for c in commits_]
    # labels_data = [c["stable"] for c in commits_]
    #
    # path_ftr = "./data/3_mar7/features.txt"
    # data_ftr = load_file(path_ftr)
    # ids_ftr, ftrs_ftr = load_ftr(data_ftr)
    # print len(ids_ftr), ftrs_ftr.shape
    #
    # ids_overlap = [i for i in ids_ftr if i in ids_data]
    # new_data = get_data(ids_overlap_=ids_overlap, ids_ftr_=ids_ftr,
    #                     data_ftr_=data_ftr, ids_data_=ids_data,
    #                     labes_data_=labels_data)
    # write_file(path_file="./data/3_mar7/new_features.txt", data=new_data)
    # print len(new_data)
    # exit()
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # path_file = "./data/3_mar7/new_features.txt"
    # balance_data_ICSE(path=path_file)
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # path_file = "./data/3_mar7/new_features_ver1.txt"
    # ids, X_, y_ = load_data_ICSE(path=path_file)
    # get_predict_ICSE(name="", X=X_, y=y_, algorithm="svm", folds=10)
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    path_ftr = "./data/3_mar7/new_features_ver1.txt"
    ids_, X_, y_ = load_data_ICSE(path=path_ftr)
    print len(ids_), X_.shape, y_.shape

    path_data = "./data/3_mar7/typediff.out"
    commits_ = extract_commit(path_file=path_data)
    nfile, nhunk, nline, nleng = 1, 8, 10, 120
    commits_ = get_commits(commits=filtering_commit(commits=commits_,
                                                    num_file=nfile,
                                                    num_hunk=nhunk,
                                                    num_loc=nline,
                                                    size_line=nleng), ids=ids_)
    X_data = create_features_ICSE(commits=commits_, ids=ids_, type="msg")
    new_X_ = np.column_stack((X_, X_data))
    print len(ids_), new_X_.shape, y_.shape
    get_predict_ICSE(name="", X=new_X_, y=y_, algorithm="svm", folds=5)
    # get_predict_ICSE(name="", X=new_X_, y=y_, algorithm="lr", folds=5)
    # get_predict_ICSE(name="", X=new_X_, y=y_, algorithm="dt", folds=5)

    # X_data = create_features_ICSE(commits=commits_, ids=ids_, type="code")
    # new_X_ = np.column_stack((X_, X_data))
    # print len(ids_), new_X_.shape, y_.shape
    # get_predict_ICSE(name="", X=new_X_, y=y_, algorithm="svm", folds=10)
    # get_predict_ICSE(name="", X=new_X_, y=y_, algorithm="lr", folds=10)
    # get_predict_ICSE(name="", X=new_X_, y=y_, algorithm="dt", folds=10)
    #
    # X_data = create_features_ICSE(commits=commits_, ids=ids_, type="all")
    # new_X_ = np.column_stack((X_, X_data))
    # print len(ids_), new_X_.shape, y_.shape
    # get_predict_ICSE(name="", X=new_X_, y=y_, algorithm="svm", folds=10)
    # get_predict_ICSE(name="", X=new_X_, y=y_, algorithm="lr", folds=10)
    # get_predict_ICSE(name="", X=new_X_, y=y_, algorithm="dt", folds=10)
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
