from sklearn.model_selection import KFold
from init_params import model_parameters, print_params
from ultis import extract_commit, filtering_commit
from baselines import extract_msg, extract_code
from data_helpers import dictionary, mapping_commit_msg, mapping_commit_code, load_label_commits
from baselines import get_items
import numpy as np
from cnn_model import CNN_model
import os
import time
import datetime

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
cntfold = 0
timestamp = str(int(time.time()))
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

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", "fold_" + str(cntfold) + "_" + timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev Summaries
            dev_summary_op = tf.summary.merge([loss_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())


            def train_step(max_msg, left_add_code, left_remove_code, left_aux_ftr,
                           right_text, right_addedcode, right_remove_code, right_aux_ftr):
                """
                A training step
                """
                feed_dict = {
                    cnn.max_msg_length: left_text,
                    cnn.input_addedcode_left: left_add_code,
                    cnn.input_removedcode_left: left_remove_code,
                    cnn.input_auxftr_left: left_aux_ftr,
                    cnn.input_text_right: right_text,
                    cnn.input_addedcode_right: right_addedcode,
                    cnn.input_removedcode_right: right_remove_code,
                    cnn.input_auxftr_right: right_aux_ftr,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }

                _, step, summaries, loss = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}".format(time_str, step, loss))
                train_summary_writer.add_summary(summaries, step)


    exit()
