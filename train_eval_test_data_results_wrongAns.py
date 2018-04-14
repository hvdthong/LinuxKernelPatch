from ultis import load_file, extract_commit_new, write_file
from baselines import filtering_commit
from data_helpers import load_label_commits, convert_to_binary
import itertools

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

    # "./data/test_data/markus_translated.out" 806
    # "./data/test_data/nicholask_translated.out" 34
    # "./data/test_data/sasha_translated.out" 1380
    test_pad_msg, test_pad_added_code, test_pad_removed_code, test_labels = list(), list(), list(), list()
    print path_test
    data = list()
    for p in path_test:
        p_data = load_file(path_file=p)
        data += p_data
    commits = extract_commit_new(commits=data)
    filter_commits = filtering_commit(commits=commits, num_file=code_file, num_hunk=code_hunk, num_loc=code_line,
                                      size_line=code_length)
    y_labels = list(convert_to_binary(load_label_commits(commits=filter_commits)))
    y_id = [c["id"] for c in filter_commits]

    print len(y_labels), len(y_id)

    path_pred = "./data/test_data_pred/fold_1523462548_model_9711.txt"
    y_pred = load_file(path_file=path_pred)
    y_pred = [float(y) for y in y_pred]

    print len(y_pred)
    print type(y_labels), type(y_id), type(y_pred)

    y_id, y_labels, y_pred = y_id[840:], y_labels[840:], y_pred[840:]
    ids = list()
    for id_, label_, pred_ in itertools.izip(y_id, y_labels, y_pred):
        if label_ != pred_:
            ids.append(id_)

    path_write = "./data/test_data/sasha_stable_pred_non.txt"
    write_file(path_file=path_write, data=ids)
