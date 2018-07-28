import tensorflow as tf


def model_parameters(path_data, msg_length, code_length, code_line, code_hunk, code_file, seed, folds,
                     embedding_dim_text, filter_sizes, ):
    # Parameters
    # ==================================================
    # Data loading
    tf.flags.DEFINE_string("path", path_data, "Loading path of our data")
    tf.flags.DEFINE_integer("msg_length", 512, "Max length of message in commits")
    tf.flags.DEFINE_integer("code_length", 120, "Max length of code in one line in commits")
    tf.flags.DEFINE_integer("code_line", 10, "Max line of code in one hunk in commits")
    tf.flags.DEFINE_integer("code_hunk", 8, "Max hunk of code in one file in commits")
    tf.flags.DEFINE_integer("code_file", 1, "Max file of code in one in commits")
