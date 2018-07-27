from ultis import extract_commit_july, load_file, write_file
from baselines_ICSE_2012 import load_ftr, load_data_ICSE, create_features_ICSE, get_predict_ICSE
import numpy as np


def getting_overlap(commits, ids_ftr, data_ftr, path_name):
    new_ftr = list()
    for c in commits:
        if c["id"] in ids_ftr:
            index_ftr = ids_ftr.index(c["id"])
            label_ftr = c["stable"]
            line_ftr = data_ftr[index_ftr] + "," + label_ftr
            new_ftr.append(line_ftr)
    write_file(path_name, new_ftr)
    return new_ftr


if __name__ == "__main__":
    # path_data = "./satisfy_typediff_sorted.out"
    # commits_ = extract_commit_july(path_file=path_data)
    # filter_commits = commits_
    #
    # ids_data = [c["id"] for c in filter_commits]
    # labels_data = [c["stable"] for c in filter_commits]
    # print len(ids_data), len(labels_data)
    #
    # path_ftr = "./features.txt"
    # data_ftr_ = load_file(path_ftr)
    # ids_ftr_, ftrs_ftr = load_ftr(data_ftr_)
    # print len(ids_ftr_), ftrs_ftr.shape
    #
    # new_ftr = getting_overlap(commits=filter_commits, ids_ftr=ids_ftr_, data_ftr=data_ftr_,
    #                           path_name="./new_features.txt")
    # print len(new_ftr)
    ###################################################################
    ###################################################################
    path_ftr = "./new_features.txt"
    ids_, X_, y_ = load_data_ICSE(path=path_ftr)
    print len(ids_), X_.shape, y_.shape

    path_data = "./satisfy_typediff_sorted.out"
    commits_ = extract_commit_july(path_file=path_data)

    X_data = create_features_ICSE(commits=commits_, ids=ids_, type="msg")
    new_X_ = np.column_stack((X_, X_data))
    print len(ids_), new_X_.shape, y_.shape
    get_predict_ICSE(name="", X=new_X_, y=y_, algorithm="svm", folds=5)
    get_predict_ICSE(name="", X=new_X_, y=y_, algorithm="lr", folds=5)
    get_predict_ICSE(name="", X=new_X_, y=y_, algorithm="dt", folds=5)
