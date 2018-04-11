from keras.models import load_model
from init_params import model_parameter_evaluation_keras, print_params
from ultis import extract_commit, filtering_commit
from baselines import extract_msg, extract_code, add_two_list
from data_helpers import dictionary, mapping_commit_msg, load_label_commits
from data_helpers import convert_to_binary
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from baselines_statistical_test import auc_score


def calculate_std(y_true, y_pred, random_state, kfold):
    print "hello"


if __name__ == "__main__":
    tf = model_parameter_evaluation_keras()
    FLAGS = tf.flags.FLAGS
    print_params(tf)

    path_file_model = "./lstm_model/"
    model_name = FLAGS.model
    # model_name = "lstm_code"
    model_name = "lstm_all"
    model = load_model(path_file_model + model_name + ".h5")

    if "msg" in FLAGS.model:
        commits_ = extract_commit(path_file=FLAGS.path)
        filter_commits = filtering_commit(commits=commits_, num_file=FLAGS.code_file, num_hunk=FLAGS.code_hunk,
                                          num_loc=FLAGS.code_line,
                                          size_line=FLAGS.code_length)
        msgs_, codes_ = extract_msg(commits=filter_commits), extract_code(commits=filter_commits)
    elif "all" in FLAGS.model:
        commits_ = extract_commit(path_file=FLAGS.path)
        filter_commits = filtering_commit(commits=commits_, num_file=FLAGS.code_file, num_hunk=FLAGS.code_hunk,
                                          num_loc=FLAGS.code_line,
                                          size_line=FLAGS.code_length)
        msgs_, codes_ = extract_msg(commits=filter_commits), extract_code(commits=filter_commits)
        all_lines = add_two_list(list1=msgs_, list2=codes_)
        msgs_ = all_lines

    elif "code" in FLAGS.model:
        commits_ = extract_commit(path_file=FLAGS.path)
        filter_commits = filtering_commit(commits=commits_, num_file=FLAGS.code_file, num_hunk=FLAGS.code_hunk,
                                          num_loc=FLAGS.code_line,
                                          size_line=FLAGS.code_length)
        msgs_, codes_ = extract_msg(commits=filter_commits), extract_code(commits=filter_commits)
        msgs_ = codes_
    else:
        print "You need to type correct model"
        exit()

    dict_msg_, dict_code_ = dictionary(data=msgs_), dictionary(data=codes_)
    pad_msg = mapping_commit_msg(msgs=msgs_, max_length=FLAGS.msg_length, dict_msg=dict_msg_)
    labels = load_label_commits(commits=filter_commits)
    labels = convert_to_binary(labels)
    print pad_msg.shape, labels.shape, labels.shape, len(dict_msg_)

    y_pred = model.predict(pad_msg, batch_size=FLAGS.batch_size)
    y_pred = np.ravel(y_pred)
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0

    accuracy = accuracy_score(y_true=labels, y_pred=y_pred)
    precision = precision_score(y_true=labels, y_pred=y_pred)
    recall = recall_score(y_true=labels, y_pred=y_pred)
    f1 = f1_score(y_true=labels, y_pred=y_pred)
    auc = auc_score(y_true=labels, y_pred=y_pred)

    print "Accuracy and std of %s: %f" % (FLAGS.model, accuracy)
    print "Precision of %s: %f" % (FLAGS.model, precision)
    print "Recall of %s: %f" % (FLAGS.model, recall)
    print "F1 of %s: %f" % (FLAGS.model, f1)
    print "AUC of %s: %f" % (FLAGS.model, auc)
