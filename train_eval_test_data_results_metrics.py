from ultis import load_file
from train_eval_test_data_results import load_commit_test_data, load_commit_train_data
from train_eval_test_data import load_id_commit_train_data
from os import listdir
from os.path import isfile, join
from ultis import load_file
from data_helpers import convert_to_binary
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from baselines_statistical_test import auc_score


def metrics_results(path_file, y_true, indexes):
    y_pred = load_file(path_file=path_file)
    y_pred = [float(i) for i in y_pred]
    accuracy, precision, recall, f1, auc = list(), list(), list(), list(), list()
    for i in range(len(indexes)):
        if i == 0:
            y_pred_, y_true_ = y_pred[:indexes[i]], y_true[:indexes[i]]
        elif i == len(indexes):
            y_pred_, y_true_ = y_pred[indexes[i]:], y_true[indexes[i]:]
        else:
            y_pred_, y_true_ = y_pred[indexes[i - 1]:indexes[i]], y_true[indexes[i - 1]:indexes[i]]
        accuracy.append(accuracy_score(y_true=y_true_, y_pred=y_pred_))
        # precision.append(precision_score(y_true=y_true_, y_pred=y_pred_))
        # recall.append(recall_score(y_true=y_true_, y_pred=y_pred_))
        # f1.append(f1_score(y_true=y_true_, y_pred=y_pred_))
        # auc.append(auc_score(y_true=y_true_, y_pred=y_pred_))
    # print len(y_true[:806])
    # print len(y_true[806:840])
    # print len(y_true[840:])
    # exit()
    # accuracy.append(accuracy_score(y_true=y_true[:806], y_pred=y_pred[:806]))
    # accuracy.append(accuracy_score(y_true=y_true[806:840], y_pred=y_pred[806:840]))
    # accuracy.append(accuracy_score(y_true=y_true[840:], y_pred=y_pred[840:]))
    print "Accuracy: ", accuracy
    # print "Precision: ", precision
    # print "Recall: ", recall
    # print "F1: ", f1
    # print "AUC: ", auc
    return accuracy


if __name__ == "__main__":
    msg_length = 512  # "Max length of message in commits"
    code_length = 120  # "Max length of code in one line in commits")
    code_line = 10  # "Max line of code in one hunk in commits")
    code_hunk = 8  # "Max hunk of code in one file in commits")
    code_file = 1  # "Max file of code in one in commits")

    path_test = list()
    path_test.append("./data/test_data/markus_translated.out")
    path_test.append("./data/test_data/nicholask_translated.out")
    path_test.append("./data/test_data/sasha_translated.out")
    test_pad_msg, test_pad_added_code, test_pad_removed_code, test_labels = list(), list(), list(), list()
    print path_test
    data = list()
    for p in path_test:
        p_data = load_file(path_file=p)
        data += p_data
    test_pad_msg, test_pad_added_code, test_pad_removed_code, test_labels, _, _ = \
        load_commit_train_data(commits=data, msg_length_=msg_length, code_length_=code_length,
                               code_line_=code_line, code_hunk_=code_hunk, code_file_=code_file)
    print test_pad_msg.shape, test_pad_added_code.shape, test_pad_removed_code.shape, test_labels.shape

    # "./data/test_data/markus_translated.out" 806
    # "./data/test_data/nicholask_translated.out" 34
    # "./data/test_data/sasha_translated.out" 1380

    indexes_ = [806, 840, 2220]
    path_pred = "./data/test_data_pred/"
    files_ = [f for f in listdir(path_pred) if isfile(join(path_pred, f))]
    print len(files_)
    for f in files_:
        print f
        metrics_results(path_file=path_pred + "/" + f, y_true=convert_to_binary(test_labels), indexes=indexes_)
        # metrics_results(path_file=path_pred + "/fold_1523462548_model_11869.txt", y_true=convert_to_binary(test_labels),
        #                 indexes=indexes_)

    ###################################################################################################################
    ###################################################################################################################
    # msg_length = 512  # "Max length of message in commits"
    # code_length = 120  # "Max length of code in one line in commits")
    # code_line = 10  # "Max line of code in one hunk in commits")
    # code_hunk = 8  # "Max hunk of code in one file in commits")
    # code_file = 1  # "Max file of code in one in commits")
    #
    # path_test = list()
    # # path_test.append("./data/test_data/markus_translated.out")
    # # path_test.append("./data/test_data/nicholask_translated.out")
    # path_test.append("./data/test_data/sasha_translated.out")
    # test_pad_msg, test_pad_added_code, test_pad_removed_code, test_labels = list(), list(), list(), list()
    # print path_test
    # data = list()
    # for p in path_test:
    #     p_data = load_file(path_file=p)
    #     data += p_data
    # ids_, test_pad_msg, test_pad_added_code, test_pad_removed_code, test_labels, _, _ = \
    #     load_id_commit_train_data(commits=data, msg_length_=msg_length, code_length_=code_length,
    #                               code_line_=code_line, code_hunk_=code_hunk, code_file_=code_file)
    # print len(ids_), test_pad_msg.shape, test_pad_added_code.shape, test_pad_removed_code.shape, test_labels.shape
    # for i, l in zip(ids_, convert_to_binary(test_labels)):
    #     print i + "\t" + str(l)
