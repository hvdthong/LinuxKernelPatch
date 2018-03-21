import tensorflow as tf


class lstm_model(object):
    def __init__(self, sequence_length, vocab_size_text, embedding_size_text,
                 hidden_size, filter_sizes, num_filters, l2_reg_lambda, num_classes):
        self.sequence_length = sequence_length
        self.vocab_size_text = vocab_size_text
        self.embedding_size_text = embedding_size_text
        self.hidden_size = hidden_size
        self.l2_reg_lambda = l2_reg_lambda
        self.num_classes = num_classes

    def _create_place_holder(self):
        # Placeholders for input, sequence length, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # Keeping track of l2 regularization loss (optional)
        self.l2_loss = tf.constant(0.0)

    def _create_embedding_layer(self):
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([self.vocab_size_text, self.embedding_size_text], -1.0, 1.0),
                trainable=True,
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
        # TODO: Embeddings process ignores commas etc. so seqlens might not be accurate for sentences with commas...


