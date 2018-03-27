from init_params import model_parameters, print_params
from eval import get_all_checkpoints
from ultis import extract_commit, filtering_commit
from data_helpers import dictionary, mapping_commit_msg, mapping_commit_code, load_label_commits
import numpy as np
from sklearn.model_selection import KFold
from baselines import extract_msg, extract_code
from baselines import get_items

def loading_data_lstm(FLAGS):
    # split data to training and testing, only load testing data
    exit()
    commits_ = extract_commit(path_file=FLAGS.path)
    filter_commits = filtering_commit(commits=commits_, num_file=FLAGS.code_file, num_hunk=FLAGS.code_hunk,
                                      num_loc=FLAGS.code_line,
                                      size_line=FLAGS.code_length)
    msgs_, codes_ = extract_msg(commits=filter_commits), extract_code(commits=filter_commits)
    dict_msg_, dict_code_ = dictionary(data=msgs_), dictionary(data=codes_)
    pad_msg = mapping_commit_msg(msgs=msgs_, max_length=FLAGS.msg_length, dict_msg=dict_msg_)
    pad_added_code = mapping_commit_code(type="added", commits=filter_commits, max_hunk=FLAGS.code_hunk,
                                         max_code_line=FLAGS.code_line,
                                         max_code_length=FLAGS.code_length, dict_code=dict_code_)
    pad_removed_code = mapping_commit_code(type="removed", commits=filter_commits, max_hunk=FLAGS.code_hunk,
                                           max_code_line=FLAGS.code_line,
                                           max_code_length=FLAGS.code_length, dict_code=dict_code_)
    labels = load_label_commits(commits=filter_commits)
    kf = KFold(n_splits=FLAGS.folds, random_state=FLAGS.seed)
    for train_index, test_index in kf.split(filter_commits):
        X_train_msg, X_test_msg = np.array(get_items(items=pad_msg, indexes=train_index)), \
                                  np.array(get_items(items=pad_msg, indexes=test_index))
        X_train_added_code, X_test_added_code = np.array(get_items(items=pad_added_code, indexes=train_index)), \
                                                np.array(get_items(items=pad_added_code, indexes=test_index))
        X_train_removed_code, X_test_removed_code = np.array(get_items(items=pad_removed_code, indexes=train_index)), \
                                                    np.array(get_items(items=pad_removed_code, indexes=test_index))
        y_train, y_test = np.array(get_items(items=labels, indexes=train_index)), \
                          np.array(get_items(items=labels, indexes=test_index))
        return X_test_msg, X_test_added_code, X_test_removed_code, y_test

if __name__ == "__main__":
    tf = model_parameters()
    FLAGS = tf.flags.FLAGS
    print_params(tf)

    if FLAGS.eval_test:
        X_test_msg, X_test_added_code, X_test_removed_code, y_test = loading_data_lstm(FLAGS=FLAGS)
        # X_test_msg, X_test_added_code, X_test_removed_code, y_test = loading_data_all(FLAGS=FLAGS)
    else:
        print "You need to turn on the evaluating file."
        exit()

    checkpoint_dir, model = "./runs/fold_0_1521641601/checkpoints", "lstm_model"
    dirs = get_all_checkpoints(checkpoint_dir=checkpoint_dir)
    dirs = [tf.train.latest_checkpoint(checkpoint_dir)]
    graph = tf.Graph()
    print dirs