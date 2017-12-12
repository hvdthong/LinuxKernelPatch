import tensorflow as tf


class CNN_model(object):
    """
    A CNN for bug fixing patches classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    Building CNN for each commit message and commit code.
    Adding extended features (optional).
    """

    # ==================================================
    # ==================================================
    def __init__(self, max_msg_length, max_code_length, max_code_line, max_code_hunk, vocab_size_text,
                 vocab_size_code, embedding_size_text, filter_sizes, num_filters, l2_reg_lambda, model):
        self.max_msg_length = max_msg_length
        self.max_code_length = max_code_length
        self.max_code_line = max_code_line
        self.max_code_hunk = max_code_hunk
        self.vocab_size_text = vocab_size_text
        self.vocab_size_code = vocab_size_code
        self.embedding_size_text = embedding_size_text
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.l2_reg_lambda = l2_reg_lambda
        self.model = model

    def _create_place_holder(self):
        # Placeholders for input and dropout
        self.input_msg = tf.placeholder(tf.int32, [None, self.max_msg_length], name='input_msg')
        self.input_addedcode = tf.placeholder(tf.int32, [None, self.max_code_hunk, self.max_code_line, self.max_code_length],
                                              name='input_addedcode')
        self.input_removedcode = tf.placeholder(tf.int32, [None, self.max_code_hunk, self.max_code_line, self.max_code_length],
                                                name='input_removedcode')
        # Label data
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")

        # loss value
        self.l2_loss = tf.constant(0.0)  # we don't use regularization in our model
        self.num_filters_total = self.num_filters * len(self.filter_sizes)

        # Create a placeholder for dropout. They depends on the model that we are going to use
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

    # ==================================================
    # ==================================================
    def _create_embedding_msg_layer(self):
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding_msg"):
            self.W_text = tf.Variable(
                tf.random_uniform([self.vocab_size_text, self.embedding_size_text], -1.0, 1.0),
                name="W_text")

    def _create_embedding_code_layer(self):
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding_code"):
            self.W_code = tf.Variable(
                tf.random_uniform([self.vocab_size_code, self.embedding_size_code], -1.0, 1.0),
                name="W_code")

    def build_graph(self):
        self._create_place_holder()
        self._create_embedding_msg_layer()
        self._create_embedding_code_layer()
