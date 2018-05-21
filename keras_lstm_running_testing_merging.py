from init_params import model_parameters, print_params
from ultis import extract_commit, filtering_commit
from baselines import extract_msg, extract_code, add_two_list
from data_helpers import dictionary, mapping_commit_msg, load_label_commits
from data_helpers import convert_to_binary
from keras_lstm import lstm_cnn, cnn_model
import numpy as np
from ultis import write_file
from baselines_statistical_test import auc_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def loading_training_data(FLAGS, type):
    if type == "msg":
        commits_ = extract_commit(path_file=FLAGS.path)
        filter_commits = filtering_commit(commits=commits_, num_file=FLAGS.code_file, num_hunk=FLAGS.code_hunk,
                                          num_loc=FLAGS.code_line,
                                          size_line=FLAGS.code_length)
        msgs_, codes_ = extract_msg(commits=filter_commits), extract_code(commits=filter_commits)
    elif type == "all":
        commits_ = extract_commit(path_file=FLAGS.path)
        filter_commits = filtering_commit(commits=commits_, num_file=FLAGS.code_file, num_hunk=FLAGS.code_hunk,
                                          num_loc=FLAGS.code_line,
                                          size_line=FLAGS.code_length)
        msgs_, codes_ = extract_msg(commits=filter_commits), extract_code(commits=filter_commits)
        all_lines = add_two_list(list1=msgs_, list2=codes_)
        msgs_ = all_lines

    elif type == "code":
        commits_ = extract_commit(path_file=FLAGS.path)
        filter_commits = filtering_commit(commits=commits_, num_file=FLAGS.code_file, num_hunk=FLAGS.code_hunk,
                                          num_loc=FLAGS.code_line,
                                          size_line=FLAGS.code_length)
        msgs_, codes_ = extract_msg(commits=filter_commits), extract_code(commits=filter_commits)
        msgs_ = codes_
    else:
        print "You need to type correct model"
        exit()
    return msgs_, codes_, filter_commits


def loading_testing_data(FLAGS, path_file, type):
    if type == "msg":
        commits_ = extract_commit(path_file=path_file)
        filter_commits = filtering_commit(commits=commits_, num_file=FLAGS.code_file, num_hunk=FLAGS.code_hunk,
                                          num_loc=FLAGS.code_line,
                                          size_line=FLAGS.code_length)
        msgs_, codes_ = extract_msg(commits=filter_commits), extract_code(commits=filter_commits)
    elif type == "all":
        commits_ = extract_commit(path_file=path_file)
        filter_commits = filtering_commit(commits=commits_, num_file=FLAGS.code_file, num_hunk=FLAGS.code_hunk,
                                          num_loc=FLAGS.code_line,
                                          size_line=FLAGS.code_length)
        msgs_, codes_ = extract_msg(commits=filter_commits), extract_code(commits=filter_commits)
        all_lines = add_two_list(list1=msgs_, list2=codes_)
        msgs_ = all_lines

    elif type == "code":
        commits_ = extract_commit(path_file=path_file)
        filter_commits = filtering_commit(commits=commits_, num_file=FLAGS.code_file, num_hunk=FLAGS.code_hunk,
                                          num_loc=FLAGS.code_line,
                                          size_line=FLAGS.code_length)
        msgs_, codes_ = extract_msg(commits=filter_commits), extract_code(commits=filter_commits)
        msgs_ = codes_
    else:
        print "You need to type correct model"
        exit()
    return msgs_, codes_, filter_commits


if __name__ == "__main__":
    tf = model_parameters()
    FLAGS = tf.flags.FLAGS
    print_params(tf)

    type = "msg"
    msgs_train, codes_train, commit_train = loading_training_data(FLAGS=FLAGS, type=type)
    path_testing_data = "./data/test_data/merging_markus_sasha.txt"
    msgs_test, codes_test, commit_test = loading_testing_data(path_file=path_testing_data,
                                                              FLAGS=FLAGS, type=type)

    msgs_train_code = msgs_train + codes_train
    dict_train = dictionary(data=msgs_train_code)
    pad_msg_train = mapping_commit_msg(msgs=msgs_train, max_length=FLAGS.msg_length, dict_msg=dict_train)
    pad_msg_test = mapping_commit_msg(msgs=msgs_test, max_length=FLAGS.msg_length, dict_msg=dict_train)
    labels_train, labels_test = load_label_commits(commits=commit_train), load_label_commits(commits=commit_test)
    labels_train, labels_test = convert_to_binary(labels_train), convert_to_binary(labels_test)
    Y_train, Y_test = labels_train, labels_test

    # name = "lstm_cnn_msg"
    # name = "lstm_cnn_code"
    # name = "lstm_cnn_all"
    # name = "cnn_msg"
    # name = "cnn_code"
    name = "cnn_all"
    if name == "lstm_cnn_msg" or name == "lstm_cnn_code" or name == "lstm_cnn_all":
        model = lstm_cnn(x_train=pad_msg_train, y_train=Y_train, x_test=pad_msg_test,
                         y_test=Y_test, dictionary_size=len(dict_train), FLAGS=FLAGS)
    elif name == "cnn_msg" or name == "cnn_code" or name == "cnn_all":
        model = cnn_model(x_train=pad_msg_train, y_train=Y_train, x_test=pad_msg_test,
                          y_test=Y_test, dictionary_size=len(dict_train), FLAGS=FLAGS)

    model.save("./lstm_model_ver3/" + name + ".h5")
    y_pred = model.predict(pad_msg_test, batch_size=FLAGS.batch_size)
    y_pred = np.ravel(y_pred)
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0

    path_file = "./data/test_data_pred_results/" + name + ".txt"
    write_file(path_file, y_pred)
    print "Accuracy of %s: %f" % (name, accuracy_score(y_true=Y_test, y_pred=y_pred))
    print "Precision of %s: %f" % (name, precision_score(y_true=Y_test, y_pred=y_pred))
    print "Recall of %s: %f" % (name, recall_score(y_true=Y_test, y_pred=y_pred))
    print "F1 of %s: %f" % (name, f1_score(y_true=Y_test, y_pred=y_pred))
    print "AUC of %s: %f" % (name, f1_score(y_true=Y_test, y_pred=y_pred))
