from ultis import load_file
from train_eval_test_data_results import load_commit_train_data
from data_helpers import convert_to_binary
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from baselines_statistical_test import auc_score
from ultis import extract_commit


def print_evaluation_metrics(y_true, y_pred):
    print "Accuracy: ", accuracy_score(y_true=y_true, y_pred=y_pred)
    print "Precision: ", precision_score(y_true=y_true, y_pred=y_pred)
    print "Recall: ", recall_score(y_true=y_true, y_pred=y_pred)
    print "F1: ", f1_score(y_true=y_true, y_pred=y_pred)
    print "AUC: ", auc_score(y_true=y_true, y_pred=y_pred)


if __name__ == "__main__":
    msg_length = 512  # "Max length of message in commits"
    code_length = 120  # "Max length of code in one line in commits")
    code_line = 10  # "Max line of code in one hunk in commits")
    code_hunk = 8  # "Max hunk of code in one file in commits")
    code_file = 1  # "Max file of code in one in commits")

    path_test = "./data/test_data/merging_markus_sasha.txt"
    data = load_file(path_file=path_test)
    test_pad_msg, test_pad_added_code, test_pad_removed_code, test_labels, _, _ = \
        load_commit_train_data(commits=data, msg_length_=msg_length, code_length_=code_length,
                               code_line_=code_line, code_hunk_=code_hunk, code_file_=code_file)
    print test_pad_msg.shape, test_pad_added_code.shape, test_pad_removed_code.shape, test_labels.shape
    labels = convert_to_binary(labels=test_labels)
    print labels

    # path_pred = "./data/test_data_pred_results/PatchNet_all.txt"
    # path_pred = "./data/test_data_pred_results/PatchNet_code.txt"
    # path_pred = "./data/test_data_pred_results/cnn_all.txt"
    # path_pred = "./data/test_data_pred_results/cnn_msg.txt"
    # path_pred = "./data/test_data_pred_results/cnn_msg_ver1.txt"
    # path_pred = "./data/test_data_pred_results/PatchNet_code.txt"
    # path_pred = "./data/test_data_pred_results/cnn_all.txt"
    # path_pred = "./data/test_data_pred_results/cnn_msg_ver1.txt"
    path_pred = "./data/test_data_pred_results/cnn_all_ver1.txt"
    path_pred = "./data/test_data_pred_results/cnn_msg_ver1.txt"
    pred = load_file(path_file=path_pred)
    pred = map(float, pred)
    print path_pred
    print_evaluation_metrics(y_true=labels, y_pred=pred)

    ##########################################################################
    ##########################################################################
    # path_data = "./data/test_data/merging_markus_sasha.txt"
    # commits_ = extract_commit(path_file=path_data)
    # labels = [1 if c["stable"] == "true" else 0 for c in commits_]
    #
    # path_pred = "./data/test_data_pred_results/bugfix.txt"
    # pred = load_file(path_file=path_pred)
    # pred = [p.split(" ")[1] for p in pred]
    # pred = map(float, pred)
    #
    # print_evaluation_metrics(y_true=labels, y_pred=pred)