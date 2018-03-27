from sklearn.model_selection import KFold
from init_params import model_parameters, print_params
from ultis import extract_commit, filtering_commit
from baselines import extract_msg, extract_code, add_two_list
from data_helpers import dictionary, mapping_commit_msg, mapping_commit_code, load_label_commits, random_mini_batch
from baselines import get_items
import numpy as np
from lstm_model import lstm_model
import os
import time
import datetime

################################################################################################
tf = model_parameters()
FLAGS = tf.flags.FLAGS
print_params(tf)
if "msg" in FLAGS.model:
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
elif "all" in FLAGS.model:
    commits_ = extract_commit(path_file=FLAGS.path)
    filter_commits = filtering_commit(commits=commits_, num_file=FLAGS.code_file, num_hunk=FLAGS.code_hunk,
                                      num_loc=FLAGS.code_line,
                                      size_line=FLAGS.code_length)
    msgs_, codes_ = extract_msg(commits=filter_commits), extract_code(commits=filter_commits)
    all_lines = add_two_list(list1=msgs_, list2=codes_)
    msgs_ = all_lines
    dict_msg_, dict_code_ = dictionary(data=msgs_), dictionary(data=codes_)
    pad_msg = mapping_commit_msg(msgs=msgs_, max_length=FLAGS.msg_length, dict_msg=dict_msg_)
    pad_added_code = mapping_commit_code(type="added", commits=filter_commits, max_hunk=FLAGS.code_hunk,
                                         max_code_line=FLAGS.code_line,
                                         max_code_length=FLAGS.code_length, dict_code=dict_code_)
    pad_removed_code = mapping_commit_code(type="removed", commits=filter_commits, max_hunk=FLAGS.code_hunk,
                                           max_code_line=FLAGS.code_line,
                                           max_code_length=FLAGS.code_length, dict_code=dict_code_)
    labels = load_label_commits(commits=filter_commits)
elif "code" in FLAGS.model:
    commits_ = extract_commit(path_file=FLAGS.path)
    filter_commits = filtering_commit(commits=commits_, num_file=FLAGS.code_file, num_hunk=FLAGS.code_hunk,
                                      num_loc=FLAGS.code_line,
                                      size_line=FLAGS.code_length)
    msgs_, codes_ = extract_msg(commits=filter_commits), extract_code(commits=filter_commits)
    msgs_ = codes_
    dict_msg_, dict_code_ = dictionary(data=msgs_), dictionary(data=codes_)
    pad_msg = mapping_commit_msg(msgs=msgs_, max_length=FLAGS.msg_length, dict_msg=dict_msg_)
    pad_added_code = mapping_commit_code(type="added", commits=filter_commits, max_hunk=FLAGS.code_hunk,
                                         max_code_line=FLAGS.code_line,
                                         max_code_length=FLAGS.code_length, dict_code=dict_code_)
    pad_removed_code = mapping_commit_code(type="removed", commits=filter_commits, max_hunk=FLAGS.code_hunk,
                                           max_code_line=FLAGS.code_line,
                                           max_code_length=FLAGS.code_length, dict_code=dict_code_)
    labels = load_label_commits(commits=filter_commits)
else:
    print "You need to type correct model"
    exit()

kf = KFold(n_splits=FLAGS.folds, random_state=FLAGS.seed)
cntfold = 0
timestamp = str(int(time.time()))

for train_index, test_index in kf.split(filter_commits):
    X_train_msg, X_test_msg = np.array(get_items(items=pad_msg, indexes=train_index)), \
                              np.array(get_items(items=pad_msg, indexes=test_index))
    X_train_added_code, X_test_added_code = np.array(get_items(items=pad_added_code, indexes=train_index)), \
                                            np.array(get_items(items=pad_added_code, indexes=test_index))
    X_train_removed_code, X_test_removed_code = np.array(get_items(items=pad_removed_code, indexes=train_index)), \
                                                np.array(get_items(items=pad_removed_code, indexes=test_index))
    y_train, y_test = np.array(get_items(items=labels, indexes=train_index)), \
                      np.array(get_items(items=labels, indexes=test_index))
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            lstm = lstm_model(
                sequence_length=FLAGS.msg_length,
                vocab_size_text=len(dict_msg_),
                embedding_size_text=FLAGS.embedding_dim_text,
                hidden_size=FLAGS.hidden_dim,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                num_classes=y_train.shape[1])
            lstm.build_graph(model=FLAGS.model)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            test_step = 0
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            grads_and_vars = optimizer.compute_gradients(lstm.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", "fold_" + str(cntfold) + "_" + timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", lstm.loss)
            acc_summary = tf.summary.scalar("accuracy", lstm.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev Summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
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


            def train_step(input_msg, input_added_code, input_removed_code, input_labels):
                """
                A training step
                """
                feed_dict = {
                    lstm.input_x: input_msg,
                    lstm.input_y: input_labels,
                    lstm.dropout_keep_prob: FLAGS.dropout_keep_prob
                }

                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, lstm.loss, lstm.accuracy],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)


            def dev_step(input_msg, input_added_code, input_removed_code, input_labels, step, len_train_batch):
                """
                A testing step
                """
                mini_batches = random_mini_batch(X_msg=input_msg, X_added_code=input_added_code,
                                                 X_removed_code=input_removed_code, Y=input_labels,
                                                 mini_batch_size=FLAGS.batch_size)
                slope = len_train_batch / float(len(mini_batches))
                accs, losses = list(), list()
                for batch in mini_batches:
                    test_input_msg, test_input_added_code, test_input_removed_code, test_input_labels = batch
                    feed_dict = {
                        lstm.input_x: test_input_msg,
                        lstm.input_y: test_input_labels,
                        lstm.dropout_keep_prob: 1.0
                    }

                    summaries, loss, accuracy = sess.run([dev_summary_op, lstm.loss,
                                                          lstm.accuracy], feed_dict)
                    accs.append(accuracy)
                    losses.append(loss)
                    if step * FLAGS.folds == 0:
                        dev_summary_writer.add_summary(summaries, 1)
                        # print "step {}".format(1)
                    else:
                        dev_summary_writer.add_summary(summaries, step * slope + 1)
                        # print "step {}".format(step * slope)
                    step += 1

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step,
                                                                sum(losses) / float(len(losses)),
                                                                sum(accs) / float(len(accs))))
                return step

        for i in xrange(0, FLAGS.num_epochs):
            # Generate batches
            mini_batches = random_mini_batch(X_msg=X_train_msg, X_added_code=X_train_added_code,
                                             X_removed_code=X_train_removed_code, Y=y_train,
                                             mini_batch_size=FLAGS.batch_size)
            saving_step = int(len(mini_batches) / 3)
            for j in xrange(len(mini_batches)):
                batch = mini_batches[j]
                input_msg, input_added_code, input_removed_code, input_labels = batch
                train_step(input_msg, input_added_code, input_removed_code, input_labels)
                current_step = tf.train.global_step(sess, global_step)
                if j == (len(mini_batches) - 1):
                    print "\nEpoch:%i" % i
                    print("\nEvaluation:")
                    test_step = dev_step(input_msg=X_test_msg, input_added_code=X_test_added_code,
                                         input_removed_code=X_test_removed_code, input_labels=y_test,
                                         step=test_step, len_train_batch=len(mini_batches))
                    print("")
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print "Saved model checkpoint to {}\n".format(path)
                elif (j + 1) % saving_step == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print "Saved model checkpoint to {}\n".format(path)
    cntfold += 1
    print_params(tf)
    exit()