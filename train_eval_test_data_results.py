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
from train_eval_test_data import load_commit_train_data, load_commit_test_data
from eval import get_all_checkpoints
from data_helpers import mini_batches, convert_to_binary
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from baselines_statistical_test import auc_score
from ultis import write_file


def get_write_name(dir):
    dir_split = dir.split("/")
    return dir_split[6] + "_" + dir_split[8].replace("-", "_")


if __name__ == "__main__":
    # checkpoint_dir = "./runs/fold_1523462548/checkpoints"
    # dirs = get_all_checkpoints(checkpoint_dir=checkpoint_dir)
    # graph = tf.Graph()
    # for d in dirs:
    #     print get_write_name(d)
    #     exit()

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
    num_epochs = 50  # "Number of training epochs (default: 200)"
    num_iters = 50000  # "Number of training iterations; the size of each iteration is the batch size " "(default: 1000)"
    evaluate_every = 100  # "Evaluate model on dev set after this many steps (default: 100)"
    checkpoint_every = 1000  # "Save model after this many steps (default: 100)"
    num_checkpoints = 1000  # "Number of checkpoints to store (default: 5)"
    # Misc Parameters
    allow_soft_placement = True  # "Allow device soft device placement"
    log_device_placement = False  # "Log placement of ops on devices"
    # Model CNN
    model = "cnn_avg_commit"  # "Running model for commit code and message"

    path_train = path
    data_train = load_file(path_file=path_train)
    train_pad_msg, train_pad_added_code, train_pad_removed_code, train_labels, dict_msg_, dict_code_ = \
        load_commit_train_data(commits=data_train, msg_length_=msg_length, code_length_=code_length,
                               code_line_=code_line, code_hunk_=code_hunk, code_file_=code_file)
    print train_pad_msg.shape, train_pad_added_code.shape, train_pad_removed_code.shape, train_labels.shape

    path_test = list()
    path_test.append("./data/test_data/markus_translated.out")
    path_test.append("./data/test_data/nicholask_translated.out")
    path_test.append("./data/test_data/sasha_translated.out")
    test_pad_msg, test_pad_added_code, test_pad_removed_code, test_labels = list(), list(), list(), list()
    print path_test
    data = list()
    for p in path_test:
        p_data = load_file(path_file=p)
        data += p_data
    test_pad_msg, test_pad_added_code, test_pad_removed_code, test_labels, _, _ = \
        load_commit_test_data(commits=data, msg_length_=msg_length, code_length_=code_length,
                              code_line_=code_line, code_hunk_=code_hunk, code_file_=code_file,
                              dict_msg_=dict_msg_, dict_code_=dict_code_)
    print test_pad_msg.shape, test_pad_added_code.shape, test_pad_removed_code.shape, test_labels.shape

    X_train_msg, X_test_msg = train_pad_msg, test_pad_msg
    X_train_added_code, X_test_added_code = train_pad_added_code, test_pad_added_code
    X_train_removed_code, X_test_removed_code = train_pad_removed_code, test_pad_removed_code
    y_train, y_test = train_labels, test_labels

    checkpoint_dir = "./runs/fold_1523462548/checkpoints"
    dirs = get_all_checkpoints(checkpoint_dir=checkpoint_dir)
    graph = tf.Graph()
    for checkpoint_file in dirs:
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=allow_soft_placement,
                log_device_placement=log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

                # Get the placeholders from the graph by name
                input_msg = graph.get_operation_by_name("input_msg").outputs[0]
                input_addedcode = graph.get_operation_by_name("input_addedcode").outputs[0]
                input_removedcode = graph.get_operation_by_name("input_removedcode").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                predictions = graph.get_operation_by_name("output/predictions").outputs[0]

                # Generate batches for one epoch
                batches = mini_batches(X_msg=X_test_msg, X_added_code=X_test_added_code,
                                       X_removed_code=X_test_removed_code,
                                       Y=y_test, mini_batch_size=batch_size)

                # Collect the predictions here
                all_predictions = []

                for batch in batches:
                    batch_input_msg, batch_input_added_code, batch_input_removed_code, batch_input_labels = batch
                    batch_predictions = sess.run(predictions,
                                                 {input_msg: batch_input_msg, input_addedcode: batch_input_added_code,
                                                  input_removedcode: batch_input_removed_code, dropout_keep_prob: 1.0})
                    all_predictions = np.concatenate([all_predictions, batch_predictions])
        path_file = "./data/test_data_pred/" + get_write_name(checkpoint_file) + ".txt"
        write_file(path_file=path_file, data=all_predictions)
        print checkpoint_file, "Accuracy:", accuracy_score(y_true=convert_to_binary(y_test), y_pred=all_predictions)
        print checkpoint_file, "Precision:", precision_score(y_true=convert_to_binary(y_test), y_pred=all_predictions)
        print checkpoint_file, "Recall:", recall_score(y_true=convert_to_binary(y_test), y_pred=all_predictions)
        print checkpoint_file, "F1:", f1_score(y_true=convert_to_binary(y_test), y_pred=all_predictions)
        print checkpoint_file, "AUC:", auc_score(y_true=convert_to_binary(y_test), y_pred=all_predictions)
        print "\n"
