from ultis import load_file, extract_commit_july, write_file
import numpy as np


def print_true_positive(id, y_pred, threshold, y_true):
    y_pred = [1 if float(y) > threshold else 0 for y in y_pred]
    true_positive = []
    for i, p, t in zip(id, y_pred, y_true):
        if p == t and t == 1:
            true_positive.append(i)
    print len(true_positive)
    path_write = "./sasha_results/true_pos_%s.txt" % (str(threshold))
    write_file(path_file=path_write, data=true_positive)


def print_false_negative(id, y_pred, threshold, y_true):
    y_pred = [1 if float(y) > threshold else 0 for y in y_pred]
    false_negative = []
    for i, p, t in zip(id, y_pred, y_true):
        if p == 0 and t == 1:
            false_negative.append(i)
    print len(false_negative)
    path_write = "./sasha_results/false_neg_%s.txt" % (str(threshold))
    write_file(path_file=path_write, data=false_negative)


if __name__ == "__main__":
    path_data = "./newres_funcalls_jul28.out.sorted.satisfy"
    commits_structure = extract_commit_july(path_file=path_data)
    commits_id = [c["id"] for c in commits_structure]

    path_true = "./statistical_test_prob_ver3/true_label.txt"
    y_true = load_file(path_file=path_true)
    y_true = [int(y) for y in y_true]

    path_pred, threshold = "./statistical_test_prob_ver3/sasha_results.txt", 50
    y_pred = load_file(path_file=path_pred)

    print_true_positive(id=commits_id, y_pred=y_pred, threshold=threshold, y_true=y_true)
    print_false_negative(id=commits_id, y_pred=y_pred, threshold=threshold, y_true=y_true)