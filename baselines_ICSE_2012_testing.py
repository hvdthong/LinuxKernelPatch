from baselines_ICSE_2012 import load_data_ICSE_new, \
    create_features_ICSE, get_predict_ICSE, load_data_ICSE, load_data_ICSE_testing_new
from ultis import load_file, extract_commit_new, get_commits, extract_commit
from baselines import filtering_commit
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from baselines import extract_msg, extract_code, add_two_list
from baselines_ICSE_2012 import get_predict_ICSE_new


def create_features_ICSE_new(commits_train, ids_train, commits_test, ids_test, type):
    new_commits_train, new_commits_test = list(), list()
    for id_ in ids_train:
        for c in commits_train:
            if c["id"] == id_:
                new_commits_train.append(c)
                break
    for id_ in ids_test:
        for c in commits_test:
            if c["id"] == id_:
                new_commits_test.append(c)
                break
    vectorizer = CountVectorizer()
    if type == "msg":
        msg_train, msg_test = extract_msg(commits=new_commits_train), extract_msg(commits=new_commits_test)
        X_train = vectorizer.fit_transform(msg_train)
        X_test = vectorizer.transform(msg_test)
    elif type == "code":
        codes_train, codes_test = extract_code(commits=new_commits_train), extract_code(commits=new_commits_test)
        X_train = vectorizer.fit_transform(codes_train)
        X_test = vectorizer.transform(codes_test)
    elif type == "all":
        msg_train, msg_test = extract_msg(commits=new_commits_train), extract_msg(commits=new_commits_test)
        codes_train, codes_test = extract_code(commits=new_commits_train), extract_code(commits=new_commits_test)
        all_lines_train = add_two_list(list1=msg_train, list2=codes_train)
        all_lines_test = add_two_list(list1=msg_test, list2=codes_test)
        X_train = vectorizer.fit_transform(all_lines_train)
        X_test = vectorizer.transform(all_lines_test)
    else:
        print "Your type is uncorrect"
        exit()
    return X_train.toarray(), X_test.toarray()


def loading_testing_data():
    data, paths = list(), list()
    # paths.append("./data/test_data/features_markusinfo.txt")
    # paths.append("./data/test_data/features_nicholaskinfo.txt")
    paths.append("./data/test_data/features_sashainfo.txt")
    for p in paths:
        data_ = load_file(path_file=p)
        data += data_
    ids_, X_ = load_data_ICSE_new(data=data)
    print len(ids_), X_.shape

    data_gt, path_gt = list(), list()
    # path_gt.append("./data/test_data/markus_translated.out")
    # path_gt.append("./data/test_data/nicholask_translated.out")
    path_gt.append("./data/test_data/sasha_translated.out")

    for p in path_gt:
        p_data = load_file(path_file=p)
        data_gt += p_data
    commits = extract_commit_new(commits=data_gt)
    nfile, nhunk, nline, nleng = 1, 8, 10, 120
    commits_ = get_commits(commits=filtering_commit(commits=commits,
                                                    num_file=nfile,
                                                    num_hunk=nhunk,
                                                    num_loc=nline,
                                                    size_line=nleng), ids=ids_)
    ids_index = [ids_.index(c["id"]) for c in commits_]
    ids_ = [ids_[i] for i in ids_index]
    X_ = X_[ids_index, :]
    y_ = [1 if c["stable"] == "true" else 0 for c in commits_]
    return commits_, ids_, X_, np.array(y_)


def loading_training_data():
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
    return commits_, ids_, X_, y_


if __name__ == "__main__":
    commits_test, ids_test, X_ftr_test, y_test = loading_testing_data()
    commits_train, ids_train, X_ftr_train, y_train = loading_training_data()
    X_msg_train, X_msg_test = create_features_ICSE_new(commits_train=commits_train, ids_train=ids_train,
                                                       commits_test=commits_test, ids_test=ids_test, type="msg")
    X_train = np.column_stack((X_ftr_train, X_msg_train))
    X_test = np.column_stack((X_ftr_test, X_msg_test))
    print X_train.shape, y_train.shape, X_test.shape, y_test.shape
    get_predict_ICSE_new(X_train=X_train, y_train=y_train,
                         X_test=X_test, y_test=y_test, algorithm="svm")
    # get_predict_ICSE_new(X_train=X_msg_train, y_train=y_train,
    #                      X_test=X_msg_test, y_test=y_test, algorithm="dt")

    # permutation = list(np.random.permutation(X_test.shape[0]))
    # shuffled_X_msg, shuffled_y_msg = X_test[permutation, :], y_test[permutation]
    # get_predict_ICSE(name="", X=X_test, y=y_test, algorithm="svm", folds=5)
