import tensorflow as tf
from tensorflow.python.ops import array_ops


class bi_LSTM_model(object):
    # ==================================================
    # ==================================================
    def __init__(self, sequence_length, vocab_size_text, embedding_size_text,
                 hidden_size, filter_sizes, num_filters, l2_reg_lambda, num_classes):
        self.sequence_length = sequence_length
        self.vocab_size_text = vocab_size_text
        self.embedding_size_text = embedding_size_text
        self.hidden_size = hidden_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.l2_reg_lambda = l2_reg_lambda
        self.num_classes = num_classes

    def _create_place_holder(self):
        # Placeholders for input, sequence length, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.seqlen = tf.placeholder(tf.int64, [None], name="seqlen")
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

    def _create_bidirectional_lstm_fw(self):
        with tf.name_scope("bidirectional-lstm-fw"):
            self.lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=1.0)

    def _create_bidirectional_lstm_bw(self):
        with tf.name_scope("bidirectional-lstm-bw"):
            self.lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=1.0)

    def _create_lstm_output_fw(self):
        with tf.variable_scope("lstm-output-fw"):
            self.lstm_outputs_fw, _ = tf.nn.dynamic_rnn(
                self.lstm_fw_cell,
                self.embedded_chars,
                sequence_length=self.seqlen,
                dtype=tf.float32)

    def _create_lstm_output_bw(self):
        with tf.variable_scope("lstm-output-bw"):
            self.embedded_chars_rev = array_ops.reverse_sequence(self.embedded_chars, seq_lengths=self.seqlen,
                                                                 seq_dim=1)
            tmp, _ = tf.nn.dynamic_rnn(
                self.lstm_bw_cell,
                self.embedded_chars_rev,
                sequence_length=self.seqlen,
                dtype=tf.float32)
            self.lstm_outputs_bw = array_ops.reverse_sequence(tmp, seq_lengths=self.seqlen, seq_dim=1)

    def _concatenate_lstm(self):
        self.lstm_outputs = tf.add(self.lstm_outputs_fw, self.lstm_outputs_bw, name="lstm_outputs")
        self.lstm_outputs_expanded = tf.expand_dims(self.lstm_outputs, -1)

    # create weight embedding layer for text and do pooling for text
    def _create_weight_conv_layer(self):
        self.w_filter_text, self.b_filter_text = [], []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.device("/cpu:" + str(filter_size)):
                with tf.name_scope("weight-conv-maxpool-text-%s" % filter_size):
                    filter_shape_text = [filter_size, self.hidden_size, 1, self.num_filters]
                    # Convolution Layer
                    w = tf.Variable(tf.truncated_normal(filter_shape_text, stddev=0.1), name="W_filter_text")
                    b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                    self.w_filter_text.append(w)
                    self.b_filter_text.append(b)

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

    def _create_conv_maxpool_2d_layer(self, filter_sizes, embedded_chars_expanded, W, b, max_msg_length):
        pooled_outputs_text = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.device("/cpu:" + str(filter_size)):
                pooled_outputs_text.append(self.pool_outputs_2d(embedded_chars_expanded=embedded_chars_expanded,
                                                                W=W[i], b=b[i], max_length=max_msg_length,
                                                                filter_size=filter_size))
        return pooled_outputs_text

    def _create_conv_maxpool_layer(self):
        pooled_outputs_text = self._create_conv_maxpool_2d_layer(filter_sizes=self.filter_sizes,
                                                                 embedded_chars_expanded=self.lstm_outputs_expanded,
                                                                 W=self.w_filter_text, b=self.b_filter_text,
                                                                 max_msg_length=self.sequence_length)
        self.pooled_outputs_text = self.h_pool_2d(num_filters_total=len(self.filter_sizes) * self.num_filters,
                                                  pooled_outputs=pooled_outputs_text)

    # ==================================================
    # adding drop_out
    def _adding_dropout_fusion_layer(self):
        self.fusion_layer_dropout = tf.nn.dropout(self.pooled_outputs_text, self.dropout_keep_prob)

    # ==================================================
    # making weight to connect fusion layer -> output layer
    def _create_weight_layer(self):
        with tf.name_scope("weight_fusion"):
            self.W_fusion = tf.get_variable(
                "W_fusion",
                shape=[self.fusion_layer_dropout.get_shape()[1], self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            self.b_fusion = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            self.l2_loss += tf.nn.l2_loss(self.W_fusion)
            self.l2_loss += tf.nn.l2_loss(self.b_fusion)

    # ==================================================
    # create output layer (score and prediction)
    def _create_output_layer(self):
        with tf.name_scope("output"):
            self.scores = tf.nn.xw_plus_b(self.fusion_layer_dropout, self.W_fusion, self.b_fusion, name="scores")
            # self.predictions = tf.sigmoid(self.scores, name="pred_prob")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

    # ==================================================
    # create output layer (score and prediction)
    def _create_loss_function(self):
        with tf.name_scope("loss"):
            # losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss

    def _measure_accuracy(self):
        with tf.name_scope("accuracy"):
            # self.pred_label = tf.to_int64(self.predictions > 0.5, name="pred_labels")
            # correct_predictions = tf.equal(self.pred_label, tf.argmax(self.input_y, 1))
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            # self.accuracy = tf.metrics.accuracy(labels=self.input_y, predictions=self.predictions)

    def build_graph(self, model):
        if "msg" in model:
            self._create_place_holder()
            self._create_embedding_layer()
            self._create_bidirectional_lstm_fw()
            self._create_bidirectional_lstm_bw()
            self._create_lstm_output_fw()
            self._create_lstm_output_bw()
            self._concatenate_lstm()
            self._create_weight_conv_layer()
            self._create_conv_maxpool_layer()
            self._adding_dropout_fusion_layer()
            self._create_weight_layer()
            self._create_output_layer()
            self._create_loss_function()
            self._measure_accuracy()