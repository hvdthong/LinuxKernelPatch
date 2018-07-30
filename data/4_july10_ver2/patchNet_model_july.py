import tensorflow as tf


class PatchNet(object):
    # ==================================================
    # ==================================================
    def __init__(self, max_msg_length, max_code_length, max_code_line, max_code_hunk, vocab_size_text,
                 vocab_size_code, embedding_size_text, filter_sizes, num_filters, l2_reg_lambda, num_classes):
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
        self.num_classes = num_classes


