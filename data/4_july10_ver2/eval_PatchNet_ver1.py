import os
import sys

split_path, goback_tokens = os.getcwd().split("/"), 2
goback_tokens = 2
path_working = "/".join(split_path[:len(split_path) - goback_tokens])
print path_working
# os.chdir(path_working + "/")
sys.path.append(path_working)

from init_params_update import model_parameters
from train_PatchNet import load_data_type
from eval import get_all_checkpoints
import numpy as np
from data_helpers import mini_batches, convert_to_binary
from eval_PatchNet import softmax
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from baselines_statistical_test import auc_score
from ultis import write_file


def eval_patchNet_train_test(tf, checkpoint_dir, test):
    FLAGS = tf.flags.FLAGS
    allow_soft_placement = True  # "Allow device soft device placement"
    log_device_placement = False  # "Log placement of ops on devices"
    dirs = get_all_checkpoints(checkpoint_dir=checkpoint_dir)
    graph = tf.Graph()

    X_test_msg, X_test_added_code, X_test_removed_code, y_test = test[0], test[1], test[2], test[3]

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
                scores = graph.get_operation_by_name("output/scores").outputs[0]

                # Generate batches for one epoch
                batches = mini_batches(X_msg=X_test_msg, X_added_code=X_test_added_code,
                                       X_removed_code=X_test_removed_code,
                                       Y=y_test, mini_batch_size=FLAGS.batch_size)

                # Collect the predictions here
                all_predictions, all_scores = [], []

                for batch in batches:
                    batch_input_msg, batch_input_added_code, batch_input_removed_code, batch_input_labels = batch
                    batch_predictions = sess.run(predictions,
                                                 {input_msg: batch_input_msg, input_addedcode: batch_input_added_code,
                                                  input_removedcode: batch_input_removed_code, dropout_keep_prob: 1.0})
                    # print batch_predictions.shape
                    all_predictions = np.concatenate([all_predictions, batch_predictions])

                    batch_scores = sess.run(scores,
                                            {input_msg: batch_input_msg, input_addedcode: batch_input_added_code,
                                             input_removedcode: batch_input_removed_code, dropout_keep_prob: 1.0})
                    batch_scores = np.ravel(softmax(batch_scores)[:, [1]])
                    # print batch_scores.shape
                    all_scores = np.concatenate([all_scores, batch_scores])
        split_checkpoint_file = checkpoint_file.split("/")
        path_write = "./patchNet_results/%s_%s.txt" % (split_checkpoint_file[-3], split_checkpoint_file[-1])
        write_file(path_file=path_write, data=all_scores)
        print checkpoint_file, "Accuracy:", accuracy_score(y_true=convert_to_binary(y_test), y_pred=all_predictions)
        print checkpoint_file, "Precision:", precision_score(y_true=convert_to_binary(y_test), y_pred=all_predictions)
        print checkpoint_file, "Recall:", recall_score(y_true=convert_to_binary(y_test), y_pred=all_predictions)
        print checkpoint_file, "F1:", f1_score(y_true=convert_to_binary(y_test), y_pred=all_predictions)
        print checkpoint_file, "AUC:", auc_score(y_true=convert_to_binary(y_test), y_pred=all_predictions)
        print "\n"


if __name__ == "__main__":
    folds_path = list()
    folds_path.append("/home/jameshoang/PycharmCode/LinuxKernelPatch/data/4_july10_ver2/runs/1534408407")
    folds_path = [f + "/checkpoints" for f in folds_path]

    num_folds_, random_state_ = 5, None
    tf_ = model_parameters(num_folds=num_folds_, random_state=random_state_, msg_length=512, code_length=120,
                           code_line=10,
                           code_hunk=8, code_file=1, embedding_dim_text=32, filter_sizes="1, 2", num_filters=32,
                           hidden_units=128, dropout_keep_prob=0.5, l2_reg_lambda=1e-5, learning_rate=1e-4,
                           batch_size=64, num_epochs=25, evaluate_every=500, checkpoint_every=1000,
                           num_checkpoints=100,
                           eval_test=False, model="all")
    FLAGS_ = tf_.flags.FLAGS
    path_, model_ = "./newres_funcalls_jul28.out.sorted.satisfy", FLAGS_.model
    print path_, model_
    pad_msg_, pad_added_code_, pad_removed_code_, labels_, dict_msg_, dict_code_ = load_data_type(path=path_,
                                                                                                  FLAGS=FLAGS_)
    print pad_msg_.shape, pad_added_code_.shape, pad_removed_code_.shape, labels_.shape
    train_data = (pad_msg_, pad_added_code_, pad_removed_code_, labels_)
    test_data = (pad_msg_, pad_added_code_, pad_removed_code_, labels_)
    for i in xrange(len(folds_path)):
        print folds_path[i]
        eval_patchNet_train_test(tf=tf_, checkpoint_dir=folds_path[i], test=test_data)
