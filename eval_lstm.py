from init_params import model_parameters, print_params
from eval import get_all_checkpoints
from ultis import extract_commit, filtering_commit
from data_helpers import dictionary, mapping_commit_msg, mapping_commit_code, \
    load_label_commits, random_mini_batch, convert_to_binary
import numpy as np
from sklearn.model_selection import KFold
from baselines import extract_msg, extract_code
from baselines import get_items
from baselines import add_two_list
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def loading_data_lstm(FLAGS):
    print FLAGS.model
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
        print X_test_msg.shape, X_test_added_code.shape, X_test_removed_code.shape, y_test.shape
        # X_test_msg, X_test_added_code, X_test_removed_code, y_test = loading_data_all(FLAGS=FLAGS)
    else:
        print "You need to turn on the evaluating file."
        exit()

    # checkpoint_dir, model = "./runs/fold_0_1521641601/checkpoints", "keras_model"
    # checkpoint_dir, model = "./runs/fold_0_1522031841/checkpoints", "lstm_all"
    checkpoint_dir, model = "./runs/fold_0_1522045240/checkpoints", "lstm_code"
    dirs = get_all_checkpoints(checkpoint_dir=checkpoint_dir)
    dirs = [tf.train.latest_checkpoint(checkpoint_dir)]
    graph = tf.Graph()
    for checkpoint_file in dirs:
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

                # Get the placeholders from the graph by name
                input_x = graph.get_operation_by_name("input_x").outputs[0]
                input_y = graph.get_operation_by_name("input_y").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                predictions = graph.get_operation_by_name("output/predictions").outputs[0]

                # Generate batches
                batches = random_mini_batch(X_msg=X_test_msg, X_added_code=X_test_added_code,
                                            X_removed_code=X_test_removed_code,
                                            Y=y_test, mini_batch_size=FLAGS.batch_size)

                # Collect the predictions here
                all_predictions = []

                for batch in batches:
                    batch_input_msg, batch_input_added_code, batch_input_removed_code, batch_input_labels = batch
                    batch_predictions = sess.run(predictions,
                                                 {input_x: batch_input_msg, input_y: batch_input_labels,
                                                  dropout_keep_prob: 1.0})
                    all_predictions = np.concatenate([all_predictions, batch_predictions])
            print checkpoint_file, "Accuracy:", accuracy_score(y_true=convert_to_binary(y_test), y_pred=all_predictions)
            print checkpoint_file, "Precision:", precision_score(y_true=convert_to_binary(y_test),
                                                                 y_pred=all_predictions)
            print checkpoint_file, "Recall:", recall_score(y_true=convert_to_binary(y_test), y_pred=all_predictions)
            print checkpoint_file, "F1:", f1_score(y_true=convert_to_binary(y_test), y_pred=all_predictions)
            print "\n"
