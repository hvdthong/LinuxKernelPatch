from sklearn.model_selection import KFold
from init_params import model_parameters, print_params
from ultis import extract_commit, filtering_commit, extract_commit_new, load_file
from baselines import extract_msg, extract_code
from data_helpers import dictionary, mapping_commit_msg, mapping_commit_code, load_label_commits, random_mini_batch
from baselines import get_items
import numpy as np
from cnn_model import CNN_model
import os
import time
import datetime
from baselines import add_two_list
import tensorflow as tf


def load_commit_data(commits, msg_length_, code_length_, code_line_, code_hunk_, code_file_):
    commits_ = extract_commit_new(commits)
    filter_commits = filtering_commit(commits=commits_, num_file=code_file, num_hunk=code_hunk,
                                      num_loc=code_line,
                                      size_line=code_length)
    msgs_, codes_ = extract_msg(commits=filter_commits), extract_code(commits=filter_commits)
    dict_msg_, dict_code_ = dictionary(data=msgs_), dictionary(data=codes_)
    pad_msg = mapping_commit_msg(msgs=msgs_, max_length=msg_length, dict_msg=dict_msg_)
    pad_added_code = mapping_commit_code(type="added", commits=filter_commits, max_hunk=code_hunk,
                                         max_code_line=code_line,
                                         max_code_length=code_length, dict_code=dict_code_)
    pad_removed_code = mapping_commit_code(type="removed", commits=filter_commits, max_hunk=code_hunk,
                                           max_code_line=code_line,
                                           max_code_length=code_length, dict_code=dict_code_)
    labels = load_label_commits(commits=filter_commits)
    return pad_msg, pad_added_code, pad_removed_code, labels, dict_msg_, dict_code_


################################################################################################
##
path = "./data/3_mar7/typediff.out"  # "Loading path of our data"
msg_length = 512  # "Max length of message in commits"
code_length = 120  # "Max length of code in one line in commits")
code_line = 10  # "Max line of code in one hunk in commits")
code_hunk = 8  # "Max hunk of code in one file in commits")
code_file = 1  # "Max file of code in one in commits")
seed = 0  # "Random seed (default:0)")
folds = 10  # "Number of folds in training data"
# Model Hyperparameters
embedding_dim_text = 64  # "Dimensionality of character embedding for text (default: 128)"
filter_sizes = "1, 2"  # Comma-separated filter sizes (default: '3,4,5')
num_filters = 16  # "Number of filters per filter size (default: 128)"
num_hidden = 100  # "Number of hidden layer units (default: 100)"
dropout_keep_prob = 0.5  # "Dropout keep probability (default: 0.5)"
l2_reg_lambda = 1e-5  # "L2 regularization lambda (default: 0.0)"
learning_rate = 1e-4  # Learning rate for optimization techniques"
# Training parameters
batch_size = 64  # "Batch Size (default: 64)")
num_epochs = 10  # "Number of training epochs (default: 200)"
num_iters = 50000  # "Number of training iterations; the size of each iteration is the batch size " "(default: 1000)"
evaluate_every = 100  # "Evaluate model on dev set after this many steps (default: 100)"
checkpoint_every = 1000  # "Save model after this many steps (default: 100)"
num_checkpoints = 1000  # "Number of checkpoints to store (default: 5)"
# Misc Parameters
allow_soft_placement = True  # "Allow device soft device placement"
log_device_placement = False  # "Log placement of ops on devices"
# Model CNN
model = "cnn_avg_commit"  # "Running model for commit code and message"

path_test = list()
path_test.append("./data/test_data/markus_translated.out")
path_test.append("./data/test_data/nicholask_translated.out")
path_test.append("./data/test_data/sasha_translated.out")
test_pad_msg, test_pad_added_code, test_pad_removed_code, test_labels = list(), list(), list(), list()
data = list()
for p in path_test:
    p_data = load_file(path_file=p)
    data += p_data
test_pad_msg, test_pad_added_code, test_pad_removed_code, test_labels, _, _ = \
    load_commit_data(commits=data, msg_length_=msg_length, code_length_=code_length,
                     code_line_=code_line, code_hunk_=code_hunk, code_file_=code_file)
print test_pad_msg.shape, test_pad_added_code.shape, test_pad_removed_code.shape, test_labels.shape

path_train = path
data_train = load_file(path_file=path_train)
train_pad_msg, train_pad_added_code, train_pad_removed_code, train_labels, dict_msg_, dict_code_ = \
    load_commit_data(commits=data_train, msg_length_=msg_length, code_length_=code_length,
                     code_line_=code_line, code_hunk_=code_hunk, code_file_=code_file)
print train_pad_msg.shape, train_pad_added_code.shape, train_pad_removed_code.shape, train_labels.shape

X_train_msg, X_test_msg = train_pad_msg, test_pad_msg
X_train_added_code, X_test_added_code = train_pad_added_code, test_pad_added_code
X_train_removed_code, X_test_removed_code = train_pad_removed_code, test_pad_removed_code
y_train, y_test = train_labels, test_labels
timestamp = str(int(time.time()))
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
                                  log_device_placement=log_device_placement)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        cnn = CNN_model(
            max_msg_length=msg_length,
            max_code_length=code_length,
            max_code_line=code_line,
            max_code_hunk=code_hunk,
            vocab_size_text=len(dict_msg_),
            vocab_size_code=len(dict_code_),
            embedding_size_text=embedding_dim_text,
            filter_sizes=list(map(int, filter_sizes.split(","))),
            num_filters=num_filters,
            l2_reg_lambda=l2_reg_lambda,
            num_classes=y_train.shape[1])
        cnn.build_graph(model=model)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        test_step = 0
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", "fold_" + timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

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
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())


        def train_step(input_msg, input_added_code, input_removed_code, input_labels):
            """
            A training step
            """
            feed_dict = {
                cnn.input_msg: input_msg,
                cnn.input_addedcode: input_added_code,
                cnn.input_removedcode: input_removed_code,
                cnn.input_y: input_labels,
                cnn.dropout_keep_prob: dropout_keep_prob
            }

            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
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
                                             mini_batch_size=batch_size)
            slope = len_train_batch / float(len(mini_batches))
            accs, losses = list(), list()
            for batch in mini_batches:
                test_input_msg, test_input_added_code, test_input_removed_code, test_input_labels = batch
                feed_dict = {
                    cnn.input_msg: test_input_msg,
                    cnn.input_addedcode: test_input_added_code,
                    cnn.input_removedcode: test_input_removed_code,
                    cnn.input_y: test_input_labels,
                    cnn.dropout_keep_prob: 1.0
                }

                summaries, loss, accuracy = sess.run([dev_summary_op, cnn.loss,
                                                      cnn.accuracy], feed_dict)
                accs.append(accuracy)
                losses.append(loss)
                if step * folds == 0:
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

    for i in xrange(0, num_epochs):
        # Generate batches
        mini_batches = random_mini_batch(X_msg=X_train_msg, X_added_code=X_train_added_code,
                                         X_removed_code=X_train_removed_code, Y=y_train,
                                         mini_batch_size=batch_size)
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
