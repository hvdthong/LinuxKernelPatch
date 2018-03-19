class lstm_model(object):
    # ==================================================
    # ==================================================
    def __init__(self, max_msg_length, max_code_length, max_code_line, max_code_hunk, vocab_size_text,
                 vocab_size_code, embedding_size_text, filter_sizes, num_filters, l2_reg_lambda, num_classes):
        print "hello"