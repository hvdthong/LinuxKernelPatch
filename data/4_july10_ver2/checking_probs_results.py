from ultis import extract_commit_july, load_file
from qualitative_analysis import load_probability_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def checking_performance(id_label, true_label, model_label, model_name):
    for i in range(5, 90, 1):
        if model_name == "patchNet":
            threshold = 1 - i / float(100)
        elif model_name == "sasha":
            threshold = 0
        threshold_label = [1 if m >= threshold else 0 for m in model_label]
        prc = precision_score(y_true=true_label, y_pred=threshold_label)
        rc = recall_score(y_true=true_label, y_pred=threshold_label)
        print threshold, prc, rc
    exit()


if __name__ == "__main__":
    path_data = "./satisfy_typediff_sorted.out"
    commits_ = extract_commit_july(path_file=path_data)
    print len(commits_), type(commits_)
    commits_id = [c["id"] for c in commits_]
    print len(commits_id)

    path_file = "./statistical_test_prob_ver2/true_label.txt"
    true_label = load_file(path_file=path_file)
    true_label = [float(t) for t in true_label]
    path_file = "./statistical_test_prob_ver2/PatchNet.txt"
    patchNet = load_file(path_file=path_file)
    patchNet = [float(t) for t in patchNet]
    checking_performance(id_label=commits_id, true_label=true_label, model_label=patchNet, model_name="patchNet")