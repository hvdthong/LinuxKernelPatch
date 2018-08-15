from init_params import model_parameters
from keras_lstm import print_params
from ultis import extract_commit_july
from sklearn.model_selection import KFold
from baselines import extract_msg, extract_code, add_two_list, get_items
from data_helpers import dictionary, mapping_commit_msg, load_label_commits
from data_helpers import convert_to_binary
import numpy as np
from keras.models import load_model
from baselines_statistical_test import make_dictionary, sorted_dict
from ultis import write_file


def loading_baseline_july(tf, folds, random_state):
    FLAGS = tf.flags.FLAGS
    commits_ = extract_commit_july(path_file=FLAGS.path)
    filter_commits = commits_
    print len(commits_)

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

    # path_file = "./statistical_test_prob/true_label.txt"
    # write_file(path_file=path_file, data=labels)
    # exit()

    print pad_msg.shape, labels.shape, len(dict_msg_)
    cntfold = 0
    pred_dict = dict()
    pred_dict_list = list()
    for i in xrange(cntfold, len(idx_folds)):
        idx = idx_folds[i]
        train_index, test_index = idx["train"], idx["test"]
        X_train_msg, X_test_msg = np.array(get_items(items=pad_msg, indexes=train_index)), \
                                  np.array(get_items(items=pad_msg, indexes=test_index))
        Y_train, Y_test = np.array(get_items(items=labels, indexes=train_index)), \
                          np.array(get_items(items=labels, indexes=test_index))
        if FLAGS.model == "lstm_cnn_all" or FLAGS.model == "lstm_cnn_msg" \
                or FLAGS.model == "lstm_cnn_code" or FLAGS.model == "cnn_all" \
                or FLAGS.model == "cnn_msg" or FLAGS.model == "cnn_code":
            path_model = "./keras_model/%s_%s.h5" % (FLAGS.model, str(cntfold))
            # path_model = "./keras_model/test_%s_%s.h5" % (FLAGS.model, str(cntfold))
            # path_model = "./keras_model/%s_%s_testing.h5" % (FLAGS.model, str(cntfold))
            model = load_model(path_model)
        else:
            print "You need to give correct model name"
            exit()
        y_pred = model.predict(X_test_msg, batch_size=FLAGS.batch_size)
        y_pred = np.ravel(y_pred)

        pred_dict.update(make_dictionary(y_pred=y_pred, y_index=test_index))

        y_pred = y_pred.tolist()
        pred_dict_list += y_pred
    # print len(pred_dict_list)
    # exit()
    # path_file = "./statistical_test_prob/" + FLAGS.model + ".txt"
    # write_file(path_file=path_file, data=sorted_dict(dict=pred_dict))
    path_file = "./statistical_test_prob/" + FLAGS.model + "_checking.txt"
    write_file(path_file=path_file, data=pred_dict_list)


if __name__ == "__main__":
    tf_ = model_parameters()
    FLAGS_ = tf_.flags.FLAGS
    print_params(tf_)

    folds_, random_state_ = 5, None
    loading_baseline_july(tf=tf_, folds=folds_, random_state=random_state_)
