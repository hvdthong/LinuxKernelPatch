# import os
# import sys
#
# split_path, goback_tokens = os.getcwd().split("/"), 2
# goback_tokens = 2
# path_working = "/".join(split_path[:len(split_path) - goback_tokens])
# print path_working
# # os.chdir(path_working + "/")
# sys.path.append(path_working)

from init_params import model_parameters
from keras_lstm import print_params
from ultis import extract_commit_july
from keras_lstm import lstm_cnn, cnn_model
from baselines import extract_msg, extract_code, add_two_list
from data_helpers import dictionary, mapping_commit_msg, load_label_commits
from data_helpers import convert_to_binary

tf_ = model_parameters()
FLAGS_ = tf_.flags.FLAGS
print_params(tf_)
FLAGS = FLAGS_

commits_ = extract_commit_july(path_file=FLAGS.path)
filter_commits = commits_

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
X_train_msg, Y_train = pad_msg, labels
X_test_msg, Y_test = pad_msg, labels

if FLAGS.model == "lstm_cnn_msg" or FLAGS.model == "lstm_cnn_code" or FLAGS.model == "lstm_cnn_all":
    model = lstm_cnn(x_train=X_train_msg, y_train=Y_train, x_test=X_test_msg,
                     y_test=Y_test, dictionary_size=len(dict_msg_), FLAGS=FLAGS)
elif FLAGS.model == "cnn_msg" or FLAGS.model == "cnn_code" or FLAGS.model == "cnn_all":
    model = cnn_model(x_train=X_train_msg, y_train=Y_train, x_test=X_test_msg,
                      y_test=Y_test, dictionary_size=len(dict_msg_), FLAGS=FLAGS)
else:
    print "You need to give correct model name"
    exit()


