from ultis import load_file, extract_commit_july, write_file
from filter_commit import get_loc_len


def msg_length(commit):
    return len(commit['msg'].split(","))


def num_code_hunk(commit):
    files = commit["code"]
    cnt_hunk = list()
    for hunk in files:
        added_hunk, removed_hunk = hunk["added"].keys(), hunk["removed"].keys()
        cnt_hunk += added_hunk + removed_hunk
    return max(cnt_hunk)


def num_code_line(commit):
    files = commit["code"]
    cnt_size_code = list()
    for hunk in files:
        removed_code, added_code = hunk["removed"], hunk["added"]
        len_loc_removed_code, len_loc_added_code = get_loc_len(removed_code), get_loc_len(added_code)
        cnt_size_code += len_loc_removed_code + len_loc_added_code
    return max(cnt_size_code)


def finding_patch_info(root_commit, commits):
    for c in commits:
        if c['id'] == root_commit:
            return root_commit + '\t' + str(msg_length(c)) + '\t' + str(num_code_hunk(c)) \
                   + '\t' + str(num_code_line(c))


if __name__ == '__main__':
    path_data = './newres_funcalls_jul28.out.sorted.satisfy'
    commits_ = extract_commit_july(path_file=path_data)
    print len(commits_)

    # path_good_commits = './statistical_test_prob_ver3/good_commits.txt'
    # good_commits = load_file(path_file=path_good_commits)
    # print len(good_commits)
    #
    # patch_good_commits = []
    # for c in good_commits:
    #     print finding_patch_info(root_commit=c, commits=commits_)
    #     patch_good_commits.append(finding_patch_info(root_commit=c, commits=commits_))
    # write_file(path_file="./statistical_test_prob_ver3/good_commits_patchInfo.txt", data=patch_good_commits)

    path_good_commits = './statistical_test_prob_ver3/bad_commits.txt'
    good_commits = load_file(path_file=path_good_commits)
    print len(good_commits)

    patch_good_commits = []
    for c in good_commits:
        print finding_patch_info(root_commit=c, commits=commits_)
        patch_good_commits.append(finding_patch_info(root_commit=c, commits=commits_))
    write_file(path_file="./statistical_test_prob_ver3/ba_commits_patchInfo.txt", data=patch_good_commits)