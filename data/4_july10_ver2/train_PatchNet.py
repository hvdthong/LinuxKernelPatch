import os
import sys

split_path, goback_tokens = os.getcwd().split("/"), 2
goback_tokens = 2
path_working = "/".join(split_path[:len(split_path) - goback_tokens])
print path_working
# os.chdir(path_working + "/")
sys.path.append(path_working)

from init_params_update import model_parameters, print_params
from ultis import extract_commit_july
from baselines import extract_msg, extract_code
from data_helpers import dictionary, mapping_commit_msg, \
    mapping_commit_code, load_label_commits, random_mini_batch
from sklearn.model_selection import KFold
import numpy as np
from baselines import get_items
from patchNet_model_july import PatchNet
import time
import datetime
from data_helpers import convert_to_binary


def load_data_type(path, FLAGS):
    commits_ = extract_commit_july(path_file=path)
    msgs_, codes_ = extract_msg(commits=commits_), extract_code(commits=commits_)
    dict_msg_, dict_code_ = dictionary(data=msgs_), dictionary(data=codes_)
    print len(commits_), len(dict_msg_), len(dict_code_)

    pad_msg = mapping_commit_msg(msgs=msgs_, max_length=FLAGS.msg_length, dict_msg=dict_msg_)
    pad_added_code = mapping_commit_code(type="added", commits=commits_, max_hunk=FLAGS.code_hunk,
                                         max_code_line=FLAGS.code_line,
                                         max_code_length=FLAGS.code_length, dict_code=dict_code_)
    pad_removed_code = mapping_commit_code(type="removed", commits=commits_, max_hunk=FLAGS.code_hunk,
                                           max_code_line=FLAGS.code_line,
                                           max_code_length=FLAGS.code_length, dict_code=dict_code_)
    labels = load_label_commits(commits=commits_)
    return pad_msg, pad_added_code, pad_removed_code, labels, dict_msg_, dict_code_


def split_train_test(data, folds, random_state):
    splits = []
    kf = KFold(n_splits=folds, random_state=random_state)
    for train_index, test_index in kf.split(data):
        splits_i = dict()
        splits_i["train"], splits_i["test"] = train_index, test_index
        splits.append(splits_i)
    return splits


def training_model_all(tf, timestamp, pad_msg, pad_added_code, pad_removed_code, labels, dict_msg,
                       dict_code):
    FLAGS = tf.flags.FLAGS

    X_train_msg, X_test_msg = pad_msg, pad_msg
    X_train_added_code, X_test_added_code = pad_added_code, pad_added_code
    X_train_removed_code, X_test_removed_code = pad_removed_code, pad_removed_code
    y_train, y_test = convert_to_binary(labels), convert_to_binary(labels)
    y_train, y_test = y_train.reshape((len(labels), 1)), y_test.reshape((len(labels), 1))
    print X_train_msg.shape, X_test_msg.shape
    print X_train_added_code.shape, X_test_added_code.shape
    print X_train_removed_code.shape, X_test_removed_code.shape
    print y_train.shape, y_test.shape

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = PatchNet(
                max_msg_length=FLAGS.msg_length,
                max_code_length=FLAGS.code_length,
                max_code_line=FLAGS.code_line,
                max_code_hunk=FLAGS.code_hunk,
                vocab_size_text=len(dict_msg),
                vocab_size_code=len(dict_code),
                embedding_size_text=FLAGS.embedding_dim_text,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                num_classes=y_train.shape[1],
                hidden_units=FLAGS.hidden_units)
            model.build_graph(model=FLAGS.model)

            # model = PatchNet(
            #     max_msg_length=FLAGS.msg_length,
            #     max_code_length=FLAGS.code_length,
            #     max_code_line=FLAGS.code_line,
            #     max_code_hunk=FLAGS.code_hunk,
            #     vocab_size_text=len(dict_msg),
            #     vocab_size_code=len(dict_code),
            #     embedding_size_text=FLAGS.embedding_dim_text,
            #     filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            #     num_filters=FLAGS.num_filters,
            #     l2_reg_lambda=FLAGS.l2_reg_lambda,
            #     num_classes=y_train.shape[1],  # note that output has only one node
            #     hidden_units=FLAGS.hidden_units)
            # model.build_graph(model=FLAGS.model)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            test_step = 0
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            grads_and_vars = optimizer.compute_gradients(model.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", "fold_" + str(fold_num) + "_" + timestamp))
            # out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp + "_" + "fold_" + str(fold_num)))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", model.loss)
            acc_summary = tf.summary.scalar("accuracy", model.accuracy)

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
                    model.input_msg: input_msg,
                    model.input_addedcode: input_added_code,
                    model.input_removedcode: input_removed_code,
                    model.input_y: input_labels,
                    model.dropout_keep_prob: FLAGS.dropout_keep_prob
                }

                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, model.loss, model.accuracy],
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
                        model.input_msg: test_input_msg,
                        model.input_addedcode: test_input_added_code,
                        model.input_removedcode: test_input_removed_code,
                        model.input_y: test_input_labels,
                        model.dropout_keep_prob: 1.0
                    }

                    summaries, loss, accuracy = sess.run([dev_summary_op, model.loss,
                                                          model.accuracy], feed_dict)
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


def training_model(tf, timestamp, fold_num, fold_index, pad_msg, pad_added_code, pad_removed_code, labels, dict_msg,
                   dict_code):
    FLAGS = tf.flags.FLAGS
    train_index, test_index = fold_index["train"], fold_index["test"]
    X_train_msg, X_test_msg = np.array(get_items(items=pad_msg, indexes=train_index)), \
                              np.array(get_items(items=pad_msg, indexes=test_index))
    X_train_added_code, X_test_added_code = np.array(get_items(items=pad_added_code, indexes=train_index)), \
                                            np.array(get_items(items=pad_added_code, indexes=test_index))
    X_train_removed_code, X_test_removed_code = np.array(get_items(items=pad_removed_code, indexes=train_index)), \
                                                np.array(get_items(items=pad_removed_code, indexes=test_index))
    # y_train, y_test = np.array(get_items(items=labels, indexes=train_index)), \
    #                   np.array(get_items(items=labels, indexes=test_index))
    y_train, y_test = convert_to_binary(labels), convert_to_binary(labels)
    y_train, y_test = y_train.reshape((len(labels), 1)), y_test.reshape((len(labels), 1))

    print X_train_msg.shape, X_test_msg.shape
    print X_train_added_code.shape, X_test_added_code.shape
    print X_train_removed_code.shape, X_test_removed_code.shape
    print y_train.shape, y_test.shape

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = PatchNet(
                max_msg_length=FLAGS.msg_length,
                max_code_length=FLAGS.code_length,
                max_code_line=FLAGS.code_line,
                max_code_hunk=FLAGS.code_hunk,
                vocab_size_text=len(dict_msg),
                vocab_size_code=len(dict_code),
                embedding_size_text=FLAGS.embedding_dim_text,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                num_classes=y_train.shape[1],
                hidden_units=FLAGS.hidden_units)
            model.build_graph(model=FLAGS.model)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            test_step = 0
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            grads_and_vars = optimizer.compute_gradients(model.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", "fold_" + str(fold_num) + "_" + timestamp))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp + "_" + "fold_" + str(fold_num)))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", model.loss)
            acc_summary = tf.summary.scalar("accuracy", model.accuracy)

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
                    model.input_msg: input_msg,
                    model.input_addedcode: input_added_code,
                    model.input_removedcode: input_removed_code,
                    model.input_y: input_labels,
                    model.dropout_keep_prob: FLAGS.dropout_keep_prob
                }

                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, model.loss, model.accuracy],
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
                        model.input_msg: test_input_msg,
                        model.input_addedcode: test_input_added_code,
                        model.input_removedcode: test_input_removed_code,
                        model.input_y: test_input_labels,
                        model.dropout_keep_prob: 1.0
                    }

                    summaries, loss, accuracy = sess.run([dev_summary_op, model.loss,
                                                          model.accuracy], feed_dict)
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


def solving_arguments(arguments):
    fold_num = int(arguments[1].split("=")[1])
    return fold_num


if __name__ == "__main__":
    print sys.argv

    # using for PatchNet
    num_folds_, random_state_ = 5, None
    tf_ = model_parameters(num_folds=num_folds_, random_state=random_state_, msg_length=512, code_length=120,
                           code_line=10,
                           code_hunk=8, code_file=1, embedding_dim_text=32, filter_sizes="1, 2", num_filters=32,
                           hidden_units=128, dropout_keep_prob=0.5, l2_reg_lambda=1e-5, learning_rate=1e-4,
                           batch_size=64, num_epochs=25, evaluate_every=500, checkpoint_every=1000,
                           num_checkpoints=100,
                           eval_test=False, model="all")
    FLAGS_ = tf_.flags.FLAGS
    # print_params(tf_)

    # path_, model_ = "./satisfy_typediff_sorted.out", FLAGS_.model
    # path_, model_ = "./satisfy_typediff_sorted_small.out", FLAGS_.model
    # path_, model_ = "./newres_funcalls_jul28.out.sorted.satisfy", FLAGS_.model
    path_, model_ = "./newres_funcalls_jul28.out.sorted.satisfy.small", FLAGS_.model
    load_data_type(path=path_, FLAGS=FLAGS_)
    print path_, model_

    pad_msg_, pad_added_code_, pad_removed_code_, labels_, dict_msg_, dict_code_ = load_data_type(path=path_,
                                                                                                  FLAGS=FLAGS_)
    print pad_msg_.shape, pad_added_code_.shape, pad_removed_code_.shape
    splits = split_train_test(data=pad_msg_, folds=num_folds_, random_state=random_state_)
    timestamp_ = str(int(time.time()))

    # fold_num_ = solving_arguments(sys.argv)
    # for i in xrange(fold_num_, len(splits)):
    #     print "Training at fold: " + str(i)
    #     training_model(tf=tf_, timestamp=timestamp_, fold_num=i, fold_index=splits[i], pad_msg=pad_msg_,
    #                    pad_added_code=pad_added_code_, pad_removed_code=pad_removed_code_, labels=labels_,
    #                    dict_msg=dict_msg_, dict_code=dict_code_)
    #     break
    # print "train fold"

    training_model_all(tf=tf_, timestamp=timestamp_, pad_msg=pad_msg_,
                       pad_added_code=pad_added_code_, pad_removed_code=pad_removed_code_, labels=labels_,
                       dict_msg=dict_msg_, dict_code=dict_code_)
    print "train all"
