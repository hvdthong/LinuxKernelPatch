from ultis import load_file, extract_commit_july


def loading_file(model):
    path_file = "./statistical_test_prob_ver3/%s.txt" % model
    y = load_file(path_file=path_file)
    return [float(v) for v in y]


def finding_false_positive(ids, y_true, y_pred, model):
    dict_fp = {}
    for id, t, p in zip(ids, y_true, y_pred):
        if model == 'sasha_results':
            if t == 0 and p > 50:
                dict_fp[id] = p
        else:
            if t == 0 and p > 0.5:
                dict_fp[id] = p

    for key, value in sorted(dict_fp.iteritems(), key=lambda (k, v): (v, k), reverse=True):
        # print "%s %s" % (key, value)
        print "%s" % (key)
    return dict_fp


if __name__ == '__main__':
    path_data = "./newres_funcalls_jul28.out.sorted.satisfy"
    commits_ = extract_commit_july(path_file=path_data)
    commits_id = [c["id"] for c in commits_]
    print len(commits_), len(commits_id)

    true_label = loading_file(model="true_label")
    # patchNet = loading_file(model="PatchNet")
    # finding_false_positive(ids=commits_id, y_true=true_label, y_pred=patchNet, model='PatchNeT')

    sasha_results = loading_file(model='sasha_results')
    finding_false_positive(ids=commits_id, y_true=true_label, y_pred=sasha_results, model='PatchNeT')
