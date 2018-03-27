import tensorflow as tf


def model_parameters():
    # Parameters
    # ==================================================
    # Data loading
    # tf.flags.DEFINE_string("path", "./data/1_oct5/eq100_line_oct5.out", "Loading path of our data")
    # tf.flags.DEFINE_string("path", "./data/1_oct5/sample_eq100_line_oct5.out", "Loading path of our data")
    tf.flags.DEFINE_string("path", "./data/3_mar7/typediff.out", "Loading path of our data")
    # tf.flags.DEFINE_string("path", "./data/3_mar7/typeaddres.out", "Loading path of our data")
    tf.flags.DEFINE_integer("msg_length", 512, "Max length of message in commits")
    tf.flags.DEFINE_integer("code_length", 120, "Max length of code in one line in commits")
    tf.flags.DEFINE_integer("code_line", 10, "Max line of code in one hunk in commits")
    tf.flags.DEFINE_integer("code_hunk", 8, "Max hunk of code in one file in commits")
    tf.flags.DEFINE_integer("code_file", 1, "Max file of code in one in commits")
    # Data training & testing
    tf.flags.DEFINE_integer("seed", 0, "Random seed (default:0)")
    tf.flags.DEFINE_integer("folds", 10, "Number of folds in training data")
    # Model Hyperparameters
    tf.flags.DEFINE_integer("embedding_dim_text", 64, "Dimensionality of character embedding for text (default: 128)")
    # tf.flags.DEFINE_integer("embedding_dim_code", 8, "Dimensionality of character embedding for code (default: 128)")
    tf.flags.DEFINE_string("filter_sizes", "1, 2, 3", "Comma-separated filter sizes (default: '3,4,5')")
    tf.flags.DEFINE_integer("num_filters", 16, "Number of filters per filter size (default: 128)")
    tf.flags.DEFINE_integer("num_hidden", 100, "Number of hidden layer units (default: 100)")
    tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
    tf.flags.DEFINE_float("l2_reg_lambda", 1e-5, "L2 regularization lambda (default: 0.0)")
    tf.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate for optimization techniques")
    # Training parameters
    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
    tf.flags.DEFINE_integer("num_iters", 50000,
                            "Number of training iterations; the size of each iteration is the batch size "
                            "(default: 1000)")
    tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
    tf.flags.DEFINE_integer("num_checkpoints", 1000, "Number of checkpoints to store (default: 5)")
    tf.flags.DEFINE_integer("num_devs", 2000, "Number of dev pairs for text and code")
    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
    # Model CNN
    # tf.flags.DEFINE_string("model", "cnn_avg_commit", "Running model for commit code and message")
    # tf.flags.DEFINE_string("model", "cnn_msg", "Running model for only commit message")
    # tf.flags.DEFINE_string("model", "cnn_code", "Running model for only commit code")
    # tf.flags.DEFINE_string("model", "cnn_msg_addedcode", "Running model for only commit added code and message")
    # Model LSTM
    tf.flags.DEFINE_integer("hidden_dim", 150, "Dimensionality of hidden layer in LSTM (default: 300")
    tf.flags.DEFINE_string("model", "lstm_msg", "Running lstm model")
    # tf.flags.DEFINE_string("model", "lstm_all", "Running lstm model")
    # tf.flags.DEFINE_string("model", "lstm_code", "Running lstm model")
    # Evaluation
    tf.flags.DEFINE_boolean("eval_test", True, "Evaluate on all testing data")
    # Qualitative Results
    tf.flags.DEFINE_boolean("qualitativeResults", True, "Evaluate on all testing data")
    return tf


def print_params(tf):
    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    print("Parameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")
