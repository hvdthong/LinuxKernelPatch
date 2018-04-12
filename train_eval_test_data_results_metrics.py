from ultis import load_file
from train_eval_test_data_results import load_commit_test_data, load_commit_train_data

if __name__ == "__main__":
    msg_length = 512  # "Max length of message in commits"
    code_length = 120  # "Max length of code in one line in commits")
    code_line = 10  # "Max line of code in one hunk in commits")
    code_hunk = 8  # "Max hunk of code in one file in commits")
    code_file = 1  # "Max file of code in one in commits")

    path_test = list()
    path_test.append("./data/test_data/markus_translated.out")
    path_test.append("./data/test_data/nicholask_translated.out")
    path_test.append("./data/test_data/sasha_translated.out")
    test_pad_msg, test_pad_added_code, test_pad_removed_code, test_labels = list(), list(), list(), list()
    print path_test
    data = list()
    for p in path_test:
        p_data = load_file(path_file=p)
        data += p_data
    test_pad_msg, test_pad_added_code, test_pad_removed_code, test_labels, _, _ = \
        load_commit_train_data(commits=data, msg_length_=msg_length, code_length_=code_length,
                               code_line_=code_line, code_hunk_=code_hunk, code_file_=code_file)
    print test_pad_msg.shape, test_pad_added_code.shape, test_pad_removed_code.shape, test_labels.shape
