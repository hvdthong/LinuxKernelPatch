from ultis import extract_commit_july, load_file, write_file
from baselines import extract_msg, extract_label, extract_code, add_two_list
from sklearn.feature_extraction.text import CountVectorizer
from qualitative_analysis import load_probability_score
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity


def load_model_labels(id):
    true_label = load_probability_score(model="true_label", threshold=None)
    patchNet = load_probability_score(model="patchNet", threshold=None)
    lstm_cnn = load_probability_score(model="LSTM-CNN", threshold=None)
    lpu_svm = load_probability_score(model="LPU-SVM", threshold=None)
    sasha = load_probability_score(model="sasha", threshold=0.3)
    dict_label = {}
    for i in xrange(len(id)):
        dict_label[id[i]] = [true_label[i], patchNet[i], lstm_cnn[i], lpu_svm[i], sasha[i]]
    return dict_label


def similarity_good_commit(id, root, all, top_k=30):
    index = id.index(root)
    root_feature = all[index, :].todense().ravel()
    similarity_score = {}
    for i in xrange(len(id)):
        compare_ftr = all[i, :].todense().ravel()
        similarity_score[id[i]] = cosine_similarity(X=root_feature, Y=compare_ftr)[0][0]
    cnt = 0
    dict_label = load_model_labels(id=id)
    write_data = list()
    for key, value in sorted(similarity_score.iteritems(), key=lambda (k, v): (v, k), reverse=True):
        if cnt >= (top_k + 1):
            break
        else:
            labels = dict_label[key]
            if labels[0] == 1 and labels[1] == 1 and labels[2] == 0 and labels[3] == 0 and labels[4] == 0:
                print key, value, cnt
                write_data.append(str(key) + "\t" + str(value) + "\t" + str(cnt))
            cnt += 1
    return write_data
    # exit()


def similarity_bad_commit(id, root, all, top_k=30):
    index = id.index(root)
    root_feature = all[index, :].todense().ravel()
    similarity_score = {}
    for i in xrange(len(id)):
        compare_ftr = all[i, :].todense().ravel()
        similarity_score[id[i]] = cosine_similarity(X=root_feature, Y=compare_ftr)[0][0]
    cnt = 0
    dict_label = load_model_labels(id=id)
    write_data = list()
    for key, value in sorted(similarity_score.iteritems(), key=lambda (k, v): (v, k), reverse=True):
        if cnt >= (top_k + 1):
            break
        else:
            labels = dict_label[key]
            if labels[0] == 1 and labels[1] == 0 and (labels[2] == 1 or labels[3] == 1 or labels[4] == 1):
                print key, value, cnt
                write_data.append(str(key) + "\t" + str(value) + "\t" + str(cnt))
            cnt += 1
    return write_data
    # exit()


if __name__ == "__main__":
    path_data = "./satisfy_typediff_sorted.out"
    commits_ = extract_commit_july(path_file=path_data)
    filter_commits = commits_
    print len(filter_commits), type(filter_commits)
    commits_id = [c["id"] for c in commits_]
    print len(commits_id)
    load_model_labels(id=commits_id)

    msgs = extract_msg(commits=filter_commits)
    labels = extract_label(commits=filter_commits)
    codes = extract_code(commits=filter_commits)
    all_lines = add_two_list(list1=msgs, list2=codes)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(all_lines)
    print X.shape

    # path_good_commits = "./statistical_test_prob_ver2/good_commits.txt"
    # good_commits = load_file(path_file=path_good_commits)
    # print "Leng of good commits: %s" % (str(len(good_commits)))
    #
    # write_data = []
    # for g in good_commits:
    #     write_data += similarity_good_commit(id=commits_id, root=g, all=X, top_k=75)
    #     # break
    # path_write = "./statistical_test_prob_ver2/good_commits_results.txt"
    # write_file(path_file=path_write, data=write_data)
    ####################################################################################
    ####################################################################################
    path_bad_commits = "./statistical_test_prob_ver2/bad_commits.txt"
    bad_commits = load_file(path_file=path_bad_commits)
    print "Leng of bad commits: %s" % (str(len(bad_commits)))

    write_data = []
    for g in bad_commits:
        write_data += similarity_bad_commit(id=commits_id, root=g, all=X, top_k=75)
        # break
    path_write = "./statistical_test_prob_ver2/bad_commits_results.txt"
    write_file(path_file=path_write, data=write_data)
