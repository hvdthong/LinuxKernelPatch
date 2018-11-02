from init_params import model_parameters
from keras_lstm import print_params
from ultis import extract_commit_july
from baselines import extract_msg, extract_code, add_two_list
from data_helpers import dictionary, mapping_commit_msg, load_label_commits
from data_helpers import convert_to_binary
import numpy as np
from baselines_statistical_test import auc_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from keras_lstm import lstm_cnn_model, cnn_network_model
from ultis import write_file


def running_epoch(epoch, pad_msg, labels, FLAGS):
    X_train_msg, X_test_msg = pad_msg, pad_msg
    Y_train, Y_test = labels, labels
    if FLAGS.model == "lstm_cnn_all" or FLAGS.model == "lstm_cnn_msg" or FLAGS.model == "lstm_cnn_code":
        path_model = "./keras_model" + "/" + FLAGS.model + "-{:02d}.hdf5".format(int(epoch))
        print path_model
        model = lstm_cnn_model(dictionary_size=len(dict_msg_), FLAGS=FLAGS)
    elif FLAGS.model == "cnn_all" or FLAGS.model == "cnn_msg" or FLAGS.model == "cnn_code":
        path_model = "./keras_model" + "/" + FLAGS.model + "-{:02d}.hdf5".format(int(epoch))
        print path_model
        model = cnn_network_model(x_train=pad_msg, y_train=labels, x_test=pad_msg,
                                  y_test=labels, dictionary_size=len(dict_msg_), FLAGS=FLAGS)
    else:
        print "Need to give the correct models"
        exit()
    model.load_weights(filepath=path_model)
    y_pred = model.predict(X_test_msg, batch_size=FLAGS.batch_size)
    y_pred = np.ravel(y_pred)
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0

    acc = accuracy_score(y_true=labels, y_pred=y_pred)
    prc = precision_score(y_true=labels, y_pred=y_pred)
    rc = recall_score(y_true=labels, y_pred=y_pred)
    f1 = f1_score(y_true=labels, y_pred=y_pred)
    auc = auc_score(y_true=labels, y_pred=y_pred)
    print acc, prc, rc, f1, auc


def print_results(epoch, pad_msg, labels, FLAGS):
    X_train_msg, X_test_msg = pad_msg, pad_msg
    Y_train, Y_test = labels, labels
    if FLAGS.model == "lstm_cnn_all" or FLAGS.model == "lstm_cnn_msg" or FLAGS.model == "lstm_cnn_code":
        path_model = "./keras_model" + "/" + FLAGS.model + "-{:02d}.hdf5".format(int(epoch))
        print path_model
        model = lstm_cnn_model(dictionary_size=len(dict_msg_), FLAGS=FLAGS)
    elif FLAGS.model == "cnn_all" or FLAGS.model == "cnn_msg" or FLAGS.model == "cnn_code":
        path_model = "./keras_model" + "/" + FLAGS.model + "-{:02d}.hdf5".format(int(epoch))
        print path_model
        model = cnn_network_model(x_train=pad_msg, y_train=labels, x_test=pad_msg,
                                  y_test=labels, dictionary_size=len(dict_msg_), FLAGS=FLAGS)
    else:
        print "Need to give the correct models"
        exit()
    model.load_weights(filepath=path_model)
    y_pred = model.predict(X_test_msg, batch_size=FLAGS.batch_size)
    y_pred = np.ravel(y_pred)

    path_write = "./statistical_test_ver2/" + FLAGS.model + "-{:02d}.hdf5".format(int(epoch)) + ".txt"
    write_file(path_file=path_write, data=y_pred)

    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0

    acc = accuracy_score(y_true=labels, y_pred=y_pred)
    prc = precision_score(y_true=labels, y_pred=y_pred)
    rc = recall_score(y_true=labels, y_pred=y_pred)
    f1 = f1_score(y_true=labels, y_pred=y_pred)
    auc = auc_score(y_true=labels, y_pred=y_pred)
    print acc, prc, rc, f1, auc


tf_ = model_parameters()
FLAGS_ = tf_.flags.FLAGS
print_params(tf_)
FLAGS = FLAGS_

commits_ = extract_commit_july(path_file=FLAGS.path)
filter_commits = commits_
print len(filter_commits)

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
num_epoch, begin_epoch = 300, 1
for i in xrange(begin_epoch, num_epoch):
    # running_epoch(epoch=i, pad_msg=pad_msg, labels=labels, FLAGS=FLAGS)
    print_results(epoch=i, pad_msg=pad_msg, labels=labels, FLAGS=FLAGS)
    # break
