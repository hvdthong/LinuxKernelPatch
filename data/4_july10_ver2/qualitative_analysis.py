from ultis import load_file, extract_commit_july, write_file
import numpy as np


def load_sasha_results(path_file, threshold):
    y_pred = load_file(path_file=path_file)
    y_pred = [float(y) for y in y_pred]
    max_value = sorted(y_pred, reverse=True)[int(len(y_pred) * (threshold - 0.05))]
    y_pred = [1 if y > max_value else 0 for y in y_pred]
    return np.array(y_pred)


def load_sasha_results_ver2(path_file, threshold):
    y_pred = load_file(path_file=path_file)
    y_pred = [float(y) for y in y_pred]
    y_pred = [1 if y > threshold else 0 for y in y_pred]
    return np.array(y_pred)


def load_probability_score(model, threshold):
    path_file = "./statistical_test_prob_ver3/%s.txt" % model
    if model == "sasha_results":
        y_pred = load_sasha_results_ver2(path_file=path_file, threshold=threshold)
    elif model == "true_label":
        y_pred = load_file(path_file=path_file)
        y_pred = np.array([float(y) for y in y_pred])
    else:
        y_pred = load_file(path_file=path_file)
        y_pred = np.array([float(y) for y in y_pred])
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    return y_pred




if __name__ == "__main__":
    # path_data = "./satisfy_typediff_sorted.out"
    path_data = "./newres_funcalls_jul28.out.sorted.satisfy"
    commits_ = extract_commit_july(path_file=path_data)
    commits_id = [c["id"] for c in commits_]
    print len(commits_), len(commits_id)

    true_label = load_probability_score(model="true_label", threshold=None)
    patchNet = load_probability_score(model="PatchNet", threshold=None)
    lstm_cnn = load_probability_score(model="LS-CNN", threshold=None)
    lpu_svm = load_probability_score(model="LPU-SVM", threshold=None)
    sasha = load_probability_score(model="sasha_results", threshold=50)
    print len(true_label), len(patchNet), len(lstm_cnn), len(lpu_svm), len(sasha)

    # # good commits can detect using patchNet
    # good_commit_id = []
    # for i in xrange(0, len(true_label)):
    #     # if true_label[i] == 1 and patchNet[i] == 1 and lstm_cnn[i] == 0 and lpu_svm[i] == 0 and sasha[i] == 0:
    #     #     good_commit_id.append(commits_id[i])
    #     if true_label[i] == 1 and patchNet[i] == 1 and sasha[i] == 0 and lstm_cnn[i] == 0 and lpu_svm[i] == 0:
    #         good_commit_id.append(commits_id[i])
    # # print len(good_commit_id)
    # path_write = "./statistical_test_prob_ver3/good_commits.txt"
    # write_file(path_file=path_write, data=good_commit_id)
    # # exit()

    bad_commit_id = []
    for i in xrange(0, len(true_label)):
        if true_label[i] == 1 and patchNet[i] == 0 and (lpu_svm[i] == 1 or sasha[i] == 1):
            bad_commit_id.append(commits_id[i])

    print len(bad_commit_id)
    # exit()

    path_write = "./statistical_test_prob_ver3/bad_commits.txt"
    write_file(path_file=path_write, data=bad_commit_id)

