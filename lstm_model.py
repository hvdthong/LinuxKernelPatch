import tensorflow as tf


class lstm_model(object):
    def __init__(self, sequence_length, vocab_size_text, embedding_size_text,
                 hidden_size, l2_reg_lambda, num_classes):
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

    def _create_LSTM_cell(self):
        lstmCell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
        lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=self.dropout_keep_prob)
        self.value, _ = tf.nn.dynamic_rnn(lstmCell, self.embedded_chars, dtype=tf.float32)

    # ==================================================
    # making weight to connect fusion layer -> output layer
    def _create_weight_layer(self):
        with tf.name_scope("weight_fusion"):
            self.W_fusion = tf.get_variable(
                "W_fusion",
                shape=[self.hidden_size, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            self.b_fusion = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            self.l2_loss += tf.nn.l2_loss(self.W_fusion)
            self.l2_loss += tf.nn.l2_loss(self.b_fusion)

    # ==================================================
    # create output layer (score and prediction)
    def _create_output_layer(self):
        with tf.name_scope("output"):
            value = tf.transpose(self.value, [1, 0, 2])
            last = tf.gather(value, int(value.get_shape()[0]) - 1)
            self.scores = tf.nn.xw_plus_b(last, self.W_fusion, self.b_fusion, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

    # ==================================================
    # create output layer (score and prediction)
    def _create_loss_function(self):
        with tf.name_scope("loss"):
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
        if "lstm" in model:
            self._create_place_holder()
            self._create_embedding_layer()
            self._create_LSTM_cell()
            self._create_weight_layer()
            self._create_output_layer()
            self._create_loss_function()
            self._measure_accuracy()
        else:
            print "Please type correct model name"
            exit()