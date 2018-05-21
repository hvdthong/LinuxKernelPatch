from baselines_ICSE_2012 import load_data_ICSE_new, \
    create_features_ICSE, get_predict_ICSE, load_data_ICSE, load_data_ICSE_testing_new
from ultis import load_file, extract_commit_new, get_commits, extract_commit
from baselines import filtering_commit
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from baselines import extract_msg, extract_code, add_two_list
from baselines_ICSE_2012 import get_predict_ICSE_new, get_predict_ICSE_writePred


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
    if type == "msg" or type == "msg_code":
        msg_train, msg_test = extract_msg(commits=new_commits_train), extract_msg(commits=new_commits_test)
        X_train = vectorizer.fit_transform(msg_train)
        X_test = vectorizer.transform(msg_test)
    else:
        print "Your type is uncorrect"
        exit()
    return X_train.toarray(), X_test.toarray()


def loading_testing_data(ftr_data, commit_data):
    ids_, X_ = load_data_ICSE_new(data=ftr_data)
    y_ = [1 if c["stable"] == "true" else 0 for c in commit_data]
    return commit_data, ids_, X_, np.array(y_)


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


def clean_merging_data(ids, ftrs):
    ftr_id = [f.split(",")[0] for f in ftrs]
    new_ftr = [ftrs[ftr_id.index(i)] for i in ids]
    return new_ftr


if __name__ == "__main__":
    path_data = "./data/test_data/merging_markus_sasha.txt"
    commits_ = extract_commit(path_file=path_data)
    nfile, nhunk, nline, nleng = 1, 8, 10, 120
    filter_commits = filtering_commit(commits=commits_, num_file=nfile, num_hunk=nhunk, num_loc=nline, size_line=nleng)
    ids_ = [c["id"] for c in filter_commits]
    labels_ = [1 if c["stable"] == "true" else 0 for c in filter_commits]

    path_ftr = "./data/test_data/features_merging_markus_sasha.txt"
    ftr = load_file(path_file=path_ftr)
    new_ftr = clean_merging_data(ids=ids_, ftrs=ftr)

    commits_test, ids_test, X_ftr_test, y_test = loading_testing_data(ftr_data=new_ftr,
                                                                      commit_data=filter_commits)
    commits_train, ids_train, X_ftr_train, y_train = loading_training_data()

    # type = "msg"
    # type = "code"
    type = "msg_code"
    print type
    # if type == "msg":
    #     X_msg_train, X_msg_test = create_features_ICSE_new(commits_train=commits_train, ids_train=ids_train,
    #                                                        commits_test=commits_test, ids_test=ids_test, type=type)
    #     X_train = X_msg_train
    #     X_test = X_msg_test
    #     print X_train.shape, y_train.shape, X_test.shape, y_test.shape
    #     path_write = "./data/test_data_pred_results/LPU_SVM_msg.txt"
    #     get_predict_ICSE_writePred(X_train=X_train, y_train=y_train,
    #                                X_test=X_test, y_test=y_test, algorithm="svm",
    #                                path_write=path_write)
    # elif type == "msg_code":
    #     X_msg_train, X_msg_test = create_features_ICSE_new(commits_train=commits_train, ids_train=ids_train,
    #                                                        commits_test=commits_test, ids_test=ids_test, type=type)
    #     X_train = np.column_stack((X_ftr_train, X_msg_train))
    #     X_test = np.column_stack((X_ftr_test, X_msg_test))
    #     print X_train.shape, y_train.shape, X_test.shape, y_test.shape
    #     path_write = "./data/test_data_pred_results/LPU_SVM_code.txt"
    #     get_predict_ICSE_writePred(X_train=X_train, y_train=y_train,
    #                                X_test=X_test, y_test=y_test, algorithm="svm",
    #                                path_write=path_write)
    # else:
    #     X_train = X_ftr_train
    #     X_test = X_ftr_test
    #     print X_train.shape, y_train.shape, X_test.shape, y_test.shape
    #     path_write = "./data/test_data_pred_results/LPU_SVM_all.txt"
    #     get_predict_ICSE_writePred(X_train=X_train, y_train=y_train,
    #                                X_test=X_test, y_test=y_test, algorithm="svm",
    #                                path_write=path_write)

    type = "msg"
    type = "code"
    type = "msg_code"
    print type
    if type == "msg":
        X_msg_train, X_msg_test = create_features_ICSE_new(commits_train=commits_train, ids_train=ids_train,
                                                           commits_test=commits_test, ids_test=ids_test, type=type)
        X_train = X_msg_train
        X_test = X_msg_test
        print X_train.shape, y_train.shape, X_test.shape, y_test.shape
        path_write = "./data/test_data_pred_results/cnn_msg_ver2.txt"
        get_predict_ICSE_writePred(X_train=X_train, y_train=y_train,
                                   X_test=X_test, y_test=y_test, algorithm="lr",
                                   path_write=path_write)
    elif type == "msg_code":
        X_msg_train, X_msg_test = create_features_ICSE_new(commits_train=commits_train, ids_train=ids_train,
                                                           commits_test=commits_test, ids_test=ids_test, type=type)
        X_train = np.column_stack((X_ftr_train, X_msg_train))
        X_test = np.column_stack((X_ftr_test, X_msg_test))
        print X_train.shape, y_train.shape, X_test.shape, y_test.shape
        path_write = "./data/test_data_pred_results/cnn_all_ver2.txt"
        get_predict_ICSE_writePred(X_train=X_train, y_train=y_train,
                                   X_test=X_test, y_test=y_test, algorithm="lr",
                                   path_write=path_write)
    else:
        X_train = X_ftr_train
        X_test = X_ftr_test
        print X_train.shape, y_train.shape, X_test.shape, y_test.shape
        path_write = "./data/test_data_pred_results/cnn_msg_ver2.txt"
        get_predict_ICSE_writePred(X_train=X_train, y_train=y_train,
                                   X_test=X_test, y_test=y_test, algorithm="lr",
                                   path_write=path_write)
