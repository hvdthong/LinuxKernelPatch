from ultis import load_file, write_file
from ultis import extract_commit, filtering_commit
from sklearn.model_selection import KFold
from baselines import get_items
from sklearn.model_selection import KFold
from random import randint
from sklearn import metrics
from scipy import stats


def get_ids_ICSE(path_file):
    data = load_file(path_file)
    new_data = list()
    for d in data:
        split_d = d.split(",")
        new_data.append(split_d[0])
    write_file(path_file="./data/3_mar7/new_features_ver1_id.txt", data=new_data)


def get_ids_data(commits, n_splits, random_state):
    kf = KFold(n_splits=n_splits, random_state=random_state)
    for train_index, test_index in kf.split(commits):
        commits_test = get_items(commits, indexes=test_index)
        test_id = [c["id"] for c in commits_test]
        test_label = [c["stable"] for c in commits_test]
        new_data = [id + "\t" + label for id, label in zip(test_id, test_label)]
        write_file(path_file="./data/3_mar7/typediff_test.out", data=new_data)
        exit()


def load_file_index(path_file, indexes):
    data = load_file(path_file=path_file)
    data = [data[i] for i in indexes]
    return data


def calculate_auc_score(y_true, y_pred, pos_label):
    fpr, tpr, thresholds = metrics.roc_curve(y_true=y_true, y_score=y_pred, pos_label=pos_label)
    return metrics.auc(fpr, tpr)


def statistical_kfold_wilcoxon(root, target, label, folds, name, type):
    if name == "normal":
        root_data = load_file(path_file=root)
        target_data = load_file(path_file=target)
        label_data = load_file(path_file=label)
        label_data = [1 if l.split("\t")[1] == "true" else 0 for l in label_data]
    elif name == "icse":
        root_data = root
        target_data = target
        label_data = label
    else:
        print "correct statement"
        exit()
    rand_num = randint(0, 99)
    kf = KFold(n_splits=folds, random_state=rand_num)
    auc_roots, auc_targets = list(), list()
    for train_index, test_index in kf.split(root_data):
        root_fold = map(float, get_items(items=root_data, indexes=test_index))
        target_fold = map(float, get_items(items=target_data, indexes=test_index))
        label_fold = map(float, get_items(items=label_data, indexes=test_index))
        if type == "auc":
            auc_roots.append(calculate_auc_score(y_true=label_fold, y_pred=root_fold, pos_label=1))
            auc_targets.append(calculate_auc_score(y_true=label_fold, y_pred=target_fold, pos_label=1))
    if type == "auc":
        _, p_value = stats.wilcoxon(auc_roots, auc_targets)
    print p_value


if __name__ == "__main__":
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # path_file = "./data/3_mar7/new_features_ver1.txt"
    # get_ids_ICSE(path_file=path_file)

    # ---------------------------------------------------------
    # path_data = "./data/3_mar7/typediff.out"
    # commits_ = extract_commit(path_file=path_data)
    # nfile, nhunk, nline, nleng = 1, 8, 10, 120
    # filter_commits = filtering_commit(commits=commits_, num_file=nfile, num_hunk=nhunk, num_loc=nline, size_line=nleng)
    # get_ids_data(commits=filter_commits, n_splits=10, random_state=0)

    # ---------------------------------------------------------
    # path_ICSE = "./data/3_mar7/new_features_ver1_id.txt"
    # icse = load_file(path_file=path_ICSE)
    # icse = [d.strip() for d in icse]
    #
    # path_data = "./data/3_mar7/typediff_test.out"
    # data = load_file(path_file=path_data)
    # data = [d.split("\t")[0] for d in data]
    #
    # intersect = set(icse).intersection(data)
    # print len(intersect)
    # ---------------------------------------------------------
    # ---------------------------------------------------------

    ## applying statistical test
    # ---------------------------------------------------------
    # path = "./statistical_test/3_mar7/"
    # root = path + "fold_0_1521433495_model-48550.txt"
    # path_label = "./data/3_mar7/typediff_test.out"
    # compare_files = list()
    # compare_files.append(path + "bi_lstm_all.txt")
    # compare_files.append(path + "bi_lstm_cnn_all.txt")
    # compare_files.append(path + "bi_lstm_cnn_code.txt")
    # compare_files.append(path + "bi_lstm_cnn_msg.txt")
    # compare_files.append(path + "bi_lstm_code.txt")
    # compare_files.append(path + "bi_lstm_msg.txt")
    # compare_files.append(path + "cnn_all.txt")
    # compare_files.append(path + "cnn_code.txt")
    # compare_files.append(path + "cnn_msg.txt")
    # compare_files.append(path + "code_dt.txt")
    # compare_files.append(path + "code_lr.txt")
    # compare_files.append(path + "code_svm.txt")
    # compare_files.append(path + "lstm_all.txt")
    # compare_files.append(path + "lstm_cnn_all.txt")
    # compare_files.append(path + "lstm_cnn_code.txt")
    # compare_files.append(path + "lstm_cnn_msg.txt")
    # compare_files.append(path + "lstm_code.txt")
    # compare_files.append(path + "lstm_msg.txt")
    # compare_files.append(path + "msg_code_dt.txt")
    # compare_files.append(path + "msg_code_lr.txt")
    # compare_files.append(path + "msg_code_svm.txt")
    # compare_files.append(path + "msg_dt.txt")
    # compare_files.append(path + "msg_lr.txt")
    # compare_files.append(path + "msg_svm.txt")
    #
    # for f in compare_files:
    #     # print root, f
    #     statistical_kfold_wilcoxon(root=root, target=f, label=path_label,
    #                                folds=25, name="normal")
        # exit()

    ## applying statistical test for ICSE'12
    # ---------------------------------------------------------
    path = "./statistical_test/3_mar7/"
    root = path + "fold_0_1521433495_model-48550.txt"
    path_label = "./data/3_mar7/typediff_test.out"

    path_ICSE = "./data/3_mar7/new_features_ver1_id.txt"
    icse = load_file(path_file=path_ICSE)
    icse = [d.strip() for d in icse]

    path_label = "./data/3_mar7/typediff_test.out"
    labels = load_file(path_file=path_label)
    labels = [d.split("\t")[0] for d in labels]

    intersect = set(icse).intersection(labels)
    intersect_icse = [icse.index(i) for i in intersect]
    intersect_data = [labels.index(i) for i in intersect]

    root_file = "./statistical_test/3_mar7/fold_0_1521433495_model-48550.txt"
    target_file = "./data/3_mar7/new_features_ver1_pred.txt"
    root_data = load_file_index(path_file=root_file, indexes=intersect_data)
    target_data = load_file_index(path_file=target_file, indexes=intersect_icse)
    label_data = load_file_index(path_file=path_label, indexes=intersect_data)
    label_data = [1 if l.split("\t")[1] == "true" else 0 for l in label_data]

    statistical_kfold_wilcoxon(root=root_data, target=target_data, label=label_data,
                               folds=20, name="icse", type="auc")
