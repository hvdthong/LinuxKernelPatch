from sklearn.model_selection import KFold
from init_params import model_parameters, print_params
from ultis import extract_commit, filtering_commit
from baselines import extract_msg, extract_code
from data_helpers import dictionary, mapping_commit_msg, mapping_commit_code, load_label_commits
from baselines import get_items
import numpy as np
from cnn_model import CNN_model

################################################################################################
tf = model_parameters()
FLAGS = tf.flags.FLAGS
print_params(tf)

commits_ = extract_commit(path_file=FLAGS.path)
filter_commits = filtering_commit(commits=commits_, num_file=FLAGS.code_file, num_hunk=FLAGS.code_hunk, num_loc=FLAGS.code_line,
                                  size_line=FLAGS.code_length)
msgs_, codes_ = extract_msg(commits=filter_commits), extract_code(commits=filter_commits)
dict_msg_, dict_code_ = dictionary(data=msgs_), dictionary(data=codes_)
pad_msg = mapping_commit_msg(msgs=msgs_, max_length=FLAGS.msg_length, dict_msg=dict_msg_)
pad_code = mapping_commit_code(commits=filter_commits, max_hunk=FLAGS.code_hunk, max_code_line=FLAGS.code_line,
                               max_code_length=FLAGS.code_length, dict_code=dict_code_)
labels = load_label_commits(commits=filter_commits)
print pad_msg.shape, pad_code.shape, labels.shape
################################################################################################
kf = KFold(n_splits=FLAGS.folds, random_state=FLAGS.seed)
for train_index, test_index in kf.split(filter_commits):
    X_train_msg, X_test_msg = np.array(get_items(items=pad_msg, indexes=train_index)), \
                              np.array(get_items(items=pad_msg, indexes=test_index))
    X_train_code, X_test_code = np.array(get_items(items=pad_code, indexes=train_index)), \
                                np.array(get_items(items=pad_code, indexes=test_index))
    y_train, y_test = np.array(get_items(items=labels, indexes=train_index)), \
                      np.array(get_items(items=labels, indexes=test_index))
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            cnn = CNN_model(
                max_msg_length=FLAGS.msg_length,
                max_code_length=FLAGS.code_length,
                max_code_line=FLAGS.code_line,
                max_code_hunk=FLAGS.code_hunk,
                vocab_size_text=len(dict_msg_),
                vocab_size_code=len(dict_code_),
                embedding_size_text=FLAGS.embedding_dim_text,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)
            cnn.build_graph(model=FLAGS.model)
    exit()
