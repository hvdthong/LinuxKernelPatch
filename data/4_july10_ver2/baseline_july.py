from ultis import extract_commit_july, filtering_commit
from baselines import extract_msg, extract_label, extract_code, add_two_list, baseline

if __name__ == "__main__":
    # path_data = "./typediff_sorted.out"
    # commits_ = extract_commit_july(path_file=path_data)
    # nfile, nhunk, nline, nleng = 1, 8, 10, 120
    # filter_commits = filtering_commit(commits=commits_, num_file=nfile, num_hunk=nhunk, num_loc=nline, size_line=nleng)
    # print len(commits_), len(filter_commits)

    path_data = "./satisfy_typediff_sorted.out"
    commits_ = extract_commit_july(path_file=path_data)
    filter_commits = commits_
    print len(commits_), len(filter_commits)

    msgs = extract_msg(commits=filter_commits)
    labels = extract_label(commits=filter_commits)
    codes = extract_code(commits=filter_commits)
    all_lines = add_two_list(list1=msgs, list2=codes)
    baseline(train=all_lines, label=labels, algorithm="svm", folds=5)
    baseline(train=all_lines, label=labels, algorithm="lr", folds=5)
    baseline(train=all_lines, label=labels, algorithm="dt", folds=5)
