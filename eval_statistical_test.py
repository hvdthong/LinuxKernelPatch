from init_params import model_parameters, print_params
from eval import loading_data
from data_helpers import mini_batches
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from data_helpers import convert_to_binary
from ultis import write_file


if __name__ == "__main__":
    tf = model_parameters()
    FLAGS = tf.flags.FLAGS
    print_params(tf)

    # CHANGE THIS: Load data. Load your own data here
    if FLAGS.eval_test:
        X_test_msg, X_test_added_code, X_test_removed_code, y_test = loading_data(FLAGS=FLAGS)
    else:
        print "You need to turn on the evaluating file."
        exit()
    checkpoint_file, model = "./runs/fold_0_1518703738/checkpoints/model-46656", "cnn_avg_commit"
    # print X_test_msg.shape, X_test_added_code.shape, X_test_removed_code.shape, y_test.shape
    split_checkpoint = checkpoint_file.split("/")
    path_file = "./statistical_test/" + split_checkpoint[2] \
                + "_" + split_checkpoint[-1] + ".txt"
    graph = tf.Graph()
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
            input_msg = graph.get_operation_by_name("input_msg").outputs[0]
            input_addedcode = graph.get_operation_by_name("input_addedcode").outputs[0]
            input_removedcode = graph.get_operation_by_name("input_removedcode").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = mini_batches(X_msg=X_test_msg, X_added_code=X_test_added_code, X_removed_code=X_test_removed_code,
                                   Y=y_test, mini_batch_size=FLAGS.batch_size)

            # Collect the predictions here
            all_predictions = []

            for batch in batches:
                batch_input_msg, batch_input_added_code, batch_input_removed_code, batch_input_labels = batch
                batch_predictions = sess.run(predictions,
                                             {input_msg: batch_input_msg, input_addedcode: batch_input_added_code,
                                              input_removedcode: batch_input_removed_code, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])
        print checkpoint_file, "Accuracy:", accuracy_score(y_true=convert_to_binary(y_test), y_pred=all_predictions)
        print checkpoint_file, "Precision:", precision_score(y_true=convert_to_binary(y_test),
                                                             y_pred=all_predictions)
        print checkpoint_file, "Recall:", recall_score(y_true=convert_to_binary(y_test), y_pred=all_predictions)
        print checkpoint_file, "F1:", f1_score(y_true=convert_to_binary(y_test), y_pred=all_predictions)
        y_pred = all_predictions
        write_file(path_file, y_pred)
        exit()