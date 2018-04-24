from ultis import load_file, write_file
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from baselines_statistical_test import auc_score


def processing_gt(data):
    ids = [d.split("\t")[0] for d in data]
    lbls = [1 if d.split("\t")[1] == "true" else 0 for d in data]
    return ids, lbls


def processing_bug_fix(data):
    ids = [d.split()[0] for d in data]
    lbls = [int(d.split()[1]) for d in data]
    return ids, lbls


def finding_index(ids_root, ids_target):
    indexes = [ids_target.index(i) for i in ids_root]
    return indexes


def finding_element(data, indexes):
    new_data = [data[i] for i in indexes]
    return new_data


def evaluation_metrics(root, target):
    print "Accuracy: %f" % (accuracy_score(y_true=root, y_pred=target))
    print "Precision: %f" % (precision_score(y_true=root, y_pred=target))
    print "Recall: %f" % (recall_score(y_true=root, y_pred=target))
    print "F1: %f" % (f1_score(y_true=root, y_pred=target))
    print "AUC: %f" % (auc_score(y_true=root, y_pred=target))


if __name__ == "__main__":
    path_gt = "./data/3_mar7/typediff_test_ver2.out"
    data_gt = load_file(path_gt)
    id_gt, lbl_gt = processing_gt(data=data_gt)

    path_bf = "./data/typediff_bug_and_fix.txt"
    data_bf = load_file(path_bf)
    id_bf, lbl_bf = processing_bug_fix(data=data_bf)
    indexes_ = finding_index(ids_root=id_gt, ids_target=id_bf)
    print len(indexes_)

    id_bf, lbl_bf = finding_element(data=id_bf, indexes=indexes_), finding_element(data=lbl_bf, indexes=indexes_)
    evaluation_metrics(root=lbl_gt, target=lbl_bf)

    # path_write = "./data/typediff_bug_and_fix_ver2.txt"
    # write_file(path_file=path_write, data=lbl_bf)
