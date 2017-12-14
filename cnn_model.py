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
        # Embedding commit message layer
        with tf.device('/cpu:0'), tf.name_scope("embedding_msg"):
            self.W_msg = tf.Variable(
                tf.random_uniform([self.vocab_size_text, self.embedding_size_text], -1.0, 1.0),
                name="W_msg")

    def _create_embedding_code_layer(self):
        # Embedding code layer
        indices = [i for i in xrange(self.vocab_size_code)]
        depth = self.vocab_size_code
        with tf.device('/cpu:0'), tf.name_scope("embedding_code"):
            self.W_code = tf.one_hot(indices=indices, depth=depth, name="W_code")

    # ==================================================
    # ==================================================
    def _create_embedding_chars_layer(self, W, input):
        embedded_chars = tf.nn.embedding_lookup(W, input)
        return tf.expand_dims(embedded_chars, -1)

    # create embedding layer for text
    def _create_embedding_chars_msg_layer(self):
        self.embedded_chars_expanded_msg = self._create_embedding_chars_layer(W=self.W_msg,
                                                                              input=self.input_msg)

    # create embedding layer for code
    def _create_embedding_chars_code_layer(self):
        self.embedded_chars_expanded_addedcode = self._create_embedding_chars_layer(W=self.W_code,
                                                                                    input=self.input_addedcode)
        self.embedded_chars_expanded_removedcode = self._create_embedding_chars_layer(W=self.W_code,
                                                                                      input=self.input_removedcode)

    # ==================================================
    # ==================================================
    # pooling for 2d metrics
    def pool_outputs_2d(self, embedded_chars_expanded, W, b, max_length, filter_size):
        conv = tf.nn.conv2d(
            embedded_chars_expanded,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")

        # Apply nonlinearity -- using elu
        h = tf.nn.elu(tf.nn.bias_add(conv, b), name="elu")

        # Maxpooling over the outputs
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, max_length - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
        return pooled

    def h_pool_2d(self, num_filters_total, pooled_outputs):
        h_pool_ = tf.reshape(tf.concat(pooled_outputs, 3), [-1, num_filters_total])
        return h_pool_

    # create weight embedding layer for text and do pooling for text
    def _create_weight_conv_msg_layer(self):
        self.w_filter_text, self.b_filter_text = [], []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.device("/cpu:" + str(filter_size)):
                with tf.name_scope("weight-conv-maxpool-text-%s" % filter_size):
                    filter_shape_text = [filter_size, self.embedding_size_text, 1, self.num_filters]
                    # Convolution Layer
                    w = tf.Variable(tf.truncated_normal(filter_shape_text, stddev=0.1), name="W_filter_text")
                    b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                    self.w_filter_text.append(w)
                    self.b_filter_text.append(b)

    def _create_conv_maxpool_2d_layer(self, filter_sizes, embedded_chars_expanded, W, b, max_msg_length):
        pooled_outputs_text = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.device("/cpu:" + str(filter_size)):
                pooled_outputs_text.append(self.pool_outputs_2d(embedded_chars_expanded=embedded_chars_expanded,
                                                                W=W[i], b=b[i], max_length=max_msg_length,
                                                                filter_size=filter_size))
        return pooled_outputs_text

    def _create_conv_maxpool_msg_layer(self):
        pooled_outputs_text = self._create_conv_maxpool_2d_layer(filter_sizes=self.filter_sizes,
                                                                 embedded_chars_expanded=self.embedded_chars_expanded_msg,
                                                                 W=self.w_filter_text, b=self.b_filter_text,
                                                                 max_msg_length=self.max_msg_length)
        self.pooled_outputs_text = self.h_pool_2d(num_filters_total=len(self.filter_sizes) * self.num_filters,
                                                  pooled_outputs=pooled_outputs_text)

    # ==================================================
    # ==================================================
    # create weight embedding layer for commit code and do pooling for commit code
    def _create_embedding_code_line(self, embedded_chars_expanded):
        embedded_chars_expanded_line = tf.reshape(tf.reduce_mean(embedded_chars_expanded, 3),
                                                  [-1, self.max_code_hunk, self.max_code_line, self.vocab_size_code])
        return embedded_chars_expanded_line

    def _create_embedding_addedcode_line(self):
        self.embedded_chars_expanded_addedcode_line = self._create_embedding_code_line(
            embedded_chars_expanded=self.embedded_chars_expanded_addedcode)

    def _create_embedding_removed_line(self):
        self.embedded_chars_expanded_removedcode_line = self._create_embedding_code_line(
            embedded_chars_expanded=self.embedded_chars_expanded_removedcode)

    def build_graph(self):
        self._create_place_holder()
        self._create_embedding_msg_layer()
        self._create_embedding_code_layer()
        self._create_embedding_chars_msg_layer()
        self._create_embedding_chars_code_layer()
        self._create_weight_conv_msg_layer()
        self._create_conv_maxpool_msg_layer()
        self._create_embedding_addedcode_line()
        self._create_embedding_removed_line()
