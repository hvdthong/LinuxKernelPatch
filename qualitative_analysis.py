from ultis import load_file, write_file
from ultis import extract_commit, filtering_commit
from baselines import extract_msg, extract_code, extract_label, add_two_list
from sklearn.feature_extraction.text import CountVectorizer
from scipy import spatial
import numpy as np


def finding_id(path_label, path_root):
    data_label = load_file(path_file=path_label)
    id_label = [d.split("\t")[0] for d in data_label]
    gt_label = [1 if d.split("\t")[1] == "true" else 0 for d in data_label]

    data_pred = load_file(path_file=path_root)
    label_pred = [float(d) for d in data_pred]

    id_correct = list()
    for i in xrange(len(id_label)):
        if gt_label[i] == label_pred[i] and gt_label[i] == 0:
            id_correct.append(id_label[i])
    return id_correct


def findining_nonoverlap(root, baselines):
    non_overlap = list()
    for id_ in root:
        if id_ not in baselines:
            non_overlap.append(id_)
    return non_overlap


def cosine_similarity(path_root, id_commit, data):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data)
    data_root = load_file(path_file=path_root)
    for id_root in data_root:
        results = list()
        index_ = id_commit.index(id_root)
        X_root = X[index_, :].toarray().flatten()
        for i in xrange(len(id_commit)):
            results.append(1 - spatial.distance.cosine(X_root, X[i, :].toarray().flatten()))
        write_file(path_file="./qualitative_analysis/cosine_sim/"
                             + id_root + ".txt", data=results)
    return None


def qualitative_looking(path_correctId, path_label):
    id_root = load_file(path_file=path_correctId)
    print len(id_root)

    data_label = load_file(path_file=path_label)
    id_label = [d.split("\t")[0] for d in data_label]
    print len(data_label)

    index_id_root = [id_label.index(i) for i in id_root]

    for id_ in id_root:
        path_id = "./qualitative_analysis/cosine_sim/" + id_ + ".txt"
        cosine_data = load_file(path_file=path_id)
        print len(cosine_data)
        cosine_data = map(float, cosine_data)
        order_cosine = sorted(cosine_data, key=float, reverse=True)
        write_data = list()
        write_data = dict()
        for jid in index_id_root:
            name_id = id_label[jid]
            cosine_score = cosine_data[jid]
            position_ = order_cosine.index(cosine_score)
            # print name_id + "\t" + str(cosine_score) + "\t" + str(position_ + 1)
            # write_data.append(name_id + "\t" + str(cosine_score) + "\t" + str(position_ + 1))
            write_data[name_id] = position_ + 1

        new_write_data = list()
        for w in sorted(write_data, key=write_data.get):
            print w, write_data[w]
            new_write_data.append(w + "\t" + str(write_data[w]))

        path_write = "./qualitative_analysis/cosine_sim_order/" + id_ + ".txt"
        write_file(path_file=path_write, data=new_write_data)


def collect_commit_mostSimilar(path_correctId):
    print "hello"


if __name__ == "__main__":
    # name_root = "fold_0_1521433495_model-48550.txt"
    # path_root_ = "./statistical_test_ver2/3_mar7/"
    # path_root = path_root_ + name_root
    #
    # path_label = "./data/3_mar7/typediff_test_ver2.out"
    # id_correct_root = finding_id(path_label=path_label, path_root=path_root)
    #
    # path_baselines = ["msg_code_dt", "lstm_all", "cnn_all",
    #                   "lstm_cnn_all", "bi_lstm_all"]
    # id_correct_baseline = list()
    # for b in path_baselines:
    #     id_b = finding_id(path_label=path_label, path_root=path_root_ + b + ".txt")
    #     id_correct_baseline += id_b
    # id_correct_baseline = set(id_correct_baseline)
    # print len(id_correct_root), len(id_correct_baseline)
    # non_overlap = findining_nonoverlap(root=id_correct_root, baselines=id_correct_baseline)
    # print len(non_overlap)
    #
    # for i in non_overlap:
    #     print i

    #############################################################################
    #############################################################################
    # path_data = "./data/3_mar7/typediff.out"
    # commits_ = extract_commit(path_file=path_data)
    # nfile, nhunk, nline, nleng = 1, 8, 10, 120
    # filter_commits = filtering_commit(commits=commits_, num_file=nfile, num_hunk=nhunk, num_loc=nline, size_line=nleng)
    # msgs = extract_msg(commits=filter_commits)
    # labels = extract_label(commits=filter_commits)
    # codes = extract_code(commits=filter_commits)
    # all_lines = add_two_list(list1=msgs, list2=codes)
    # ids_commits = [c["id"] for c in filter_commits]
    # print len(ids_commits)
    #
    # path_root_id = "./qualitative_analysis/correct_PatchNet_nonCorrect_baselines.txt"
    # cosine_similarity(path_root=path_root_id, id_commit=ids_commits, data=all_lines)
    #############################################################################
    #############################################################################
    # path_correctId = "./qualitative_analysis/correct_PatchNet_nonCorrect_baselines.txt"
    # path_label = "./data/3_mar7/typediff_test_ver2.out"
    #
    # qualitative_looking(path_correctId=path_correctId, path_label=path_label)
    #############################################################################
    #############################################################################
    path_correctId = "./qualitative_analysis/correct_PatchNet_nonCorrect_baselines.txt"
    collect_commit_mostSimilar(path_correctId=path_correctId)