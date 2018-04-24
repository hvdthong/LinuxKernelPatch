from ultis import extract_commit, filtering_commit, load_file
from train_eval_test_data import load_commit_train_data
from baselines import extract_msg, extract_code, add_two_list
from data_helpers import mapping_commit_msg, load_label_commits
from keras.models import load_model
from data_helpers import convert_to_binary
import numpy as np
from sklearn.metrics import accuracy_score
from data_helpers import dictionary

if __name__ == "__main__":
    msg_length = 512  # "Max length of message in commits"
    code_length = 120  # "Max length of code in one line in commits")
    code_line = 10  # "Max line of code in one hunk in commits")
    code_hunk = 8  # "Max hunk of code in one file in commits")
    code_file = 1  # "Max file of code in one in commits")

    path_train = "./data/3_mar7/typediff.out"
    # data_train = load_file(path_file=path_train)
    # train_pad_msg, train_pad_added_code, train_pad_removed_code, train_labels, dict_msg_, dict_code_ = \
    #     load_commit_train_data(commits=data_train, msg_length_=msg_length, code_length_=code_length,
    #                            code_line_=code_line, code_hunk_=code_hunk, code_file_=code_file)
    # print train_pad_msg.shape, train_pad_added_code.shape, train_pad_removed_code.shape, train_labels.shape

    commits_train = extract_commit(path_file=path_train)
    filter_commits_train = filtering_commit(commits=commits_train, num_file=code_file, num_hunk=code_hunk,
                                            num_loc=code_line,
                                            size_line=code_length)
    msgs_train, codes_train = extract_msg(commits=filter_commits_train), extract_code(commits=filter_commits_train)
    dict_msg_train, dict_code_train = dictionary(data=msgs_train), dictionary(data=codes_train)

    path_test = "./data/test_data/markus_translated.out"
    # path_test = "./data/test_data/sasha_translated.out"
    commits_test = extract_commit(path_file=path_test)
    filter_commits_test = filtering_commit(commits=commits_test, num_file=code_file, num_hunk=code_hunk,
                                           num_loc=code_line,
                                           size_line=code_length)
    msgs_test, codes_test = extract_msg(commits=filter_commits_test), extract_code(commits=filter_commits_test)
    all_lines_test = add_two_list(list1=msgs_test, list2=codes_test)
    msgs_test = msgs_test

    dict_msg_ = dict_msg_train.update(dict_code_train)
    pad_msg_test = mapping_commit_msg(msgs=msgs_test, max_length=msg_length, dict_msg=dict_msg_)
    labels = load_label_commits(commits=filter_commits_test)
    labels = convert_to_binary(labels)

    model_name = "cnn_all"
    model_name = "lstm_all"
    model_name = "bi_lstm_all"
    model_name = "lstm_cnn_all"
    print path_test, model_name
    model_path = "./lstm_model_ver2/" + model_name + "_0.h5"
    model = load_model(model_path)
    y_pred = model.predict(pad_msg_test, batch_size=32)
    y_pred = np.ravel(y_pred)
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    print accuracy_score(y_true=labels, y_pred=y_pred)
