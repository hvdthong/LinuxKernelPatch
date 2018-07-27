import sys
# sys_path = "/home/jameshoang/PycharmCode/LinuxKernelPatch/"
sys_path = "/home/thonghoang/PycharmCode/LinuxKernelPatch/"
sys.path.append(sys_path)
from init_params import model_parameters
from keras_lstm import print_params
from ultis import extract_commit_july, write_file
from baselines import extract_msg, extract_code, add_two_list, get_items
from data_helpers import dictionary, mapping_commit_msg, load_label_commits
from data_helpers import convert_to_binary
import time
from sklearn.model_selection import KFold
import numpy as np
from keras_lstm import lstm_cnn, cnn_model
from baselines_statistical_test import auc_score, make_dictionary, sorted_dict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def running_baseline_july(tf, folds, random_state):
    commits_ = extract_commit_july(path_file=FLAGS.path)
    filter_commits = commits_
    kf = KFold(n_splits=folds, random_state=random_state)
    idx_folds = list()
    for train_index, test_index in kf.split(filter_commits):
        idx = dict()
        idx["train"], idx["test"] = train_index, test_index
        idx_folds.append(idx)

    if "msg" in FLAGS.model:
        msgs_, codes_ = extract_msg(commits=filter_commits), extract_code(commits=filter_commits)
    elif "all" in FLAGS.model:
        msgs_, codes_ = extract_msg(commits=filter_commits), extract_code(commits=filter_commits)
        all_lines = add_two_list(list1=msgs_, list2=codes_)
        msgs_ = all_lines
    elif "code" in FLAGS.model:
        msgs_, codes_ = extract_msg(commits=filter_commits), extract_code(commits=filter_commits)
        msgs_ = codes_
    else:
        print "You need to type correct model"
        exit()

    dict_msg_, dict_code_ = dictionary(data=msgs_), dictionary(data=codes_)
    pad_msg = mapping_commit_msg(msgs=msgs_, max_length=FLAGS.msg_length, dict_msg=dict_msg_)
    labels = load_label_commits(commits=filter_commits)
    labels = convert_to_binary(labels)
    print pad_msg.shape, labels.shape, len(dict_msg_)

    timestamp = str(int(time.time()))
    accuracy, precision, recall, f1, auc = list(), list(), list(), list(), list()
    cntfold = 0
    pred_dict = dict()
    for idx in idx_folds:
        train_index, test_index = idx["train"], idx["test"]
        X_train_msg, X_test_msg = np.array(get_items(items=pad_msg, indexes=train_index)), \
                                  np.array(get_items(items=pad_msg, indexes=test_index))
        Y_train, Y_test = np.array(get_items(items=labels, indexes=train_index)), \
                          np.array(get_items(items=labels, indexes=test_index))
        if FLAGS.model == "lstm_cnn_msg" or FLAGS.model == "lstm_cnn_code" or FLAGS.model == "lstm_cnn_all":
            model = lstm_cnn(x_train=X_train_msg, y_train=Y_train, x_test=X_test_msg,
                             y_test=Y_test, dictionary_size=len(dict_msg_), FLAGS=FLAGS)
        elif FLAGS.model == "cnn_msg" or FLAGS.model == "cnn_code" or FLAGS.model == "cnn_all":
            model = cnn_model(x_train=X_train_msg, y_train=Y_train, x_test=X_test_msg,
                              y_test=Y_test, dictionary_size=len(dict_msg_), FLAGS=FLAGS)
        else:
            print "You need to give correct model name"
            exit()

        model.save("./lstm_model/" + FLAGS.model + "_" + str(cntfold) + ".h5")
        cntfold += 1
        y_pred = model.predict(X_test_msg, batch_size=FLAGS.batch_size)
        y_pred = np.ravel(y_pred)
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
        pred_dict.update(make_dictionary(y_pred=y_pred, y_index=test_index))
        accuracy.append(accuracy_score(y_true=Y_test, y_pred=y_pred))
        precision.append(precision_score(y_true=Y_test, y_pred=y_pred))
        recall.append(recall_score(y_true=Y_test, y_pred=y_pred))
        f1.append(f1_score(y_true=Y_test, y_pred=y_pred))
        auc.append(auc_score(y_true=Y_test, y_pred=y_pred))
    path_file = "./statistical_test/" + FLAGS.model + ".txt"
    write_file(path_file=path_file, data=sorted_dict(dict=pred_dict))
    print accuracy, "Accuracy and std of %s: %f %f" % (
        FLAGS.model, np.mean(np.array(accuracy)), np.std(np.array(accuracy)))
    print precision, "Precision of %s: %f %f" % (FLAGS.model, np.mean(np.array(precision)), np.std(np.array(precision)))
    print recall, "Recall of %s: %f %f" % (FLAGS.model, np.mean(np.array(recall)), np.std(np.array(recall)))
    print f1, "F1 of %s: %f %f" % (FLAGS.model, np.mean(np.array(f1)), np.std(np.array(f1)))
    print auc, "AUC of %s: %f %f" % (FLAGS.model, np.mean(np.array(auc)), np.std(np.array(auc)))
    print_params(tf)


if __name__ == "__main__":
    tf_ = model_parameters()
    FLAGS = tf_.flags.FLAGS
    print_params(tf_)

    folds_, random_state_ = 5, None
    running_baseline_july(tf=tf_, folds=folds_, random_state=random_state_)
