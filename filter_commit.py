def filter_number_code_file(commits, num_file):
    # check the number of code file in the commit code
    commit_id = list()
    for c in commits:
        if len(c["code"]) <= num_file:
            commit_id.append(c["id"])
    return commit_id


def filter_number_code_hunk(commits, num_hunk):
    commit_id = list()
    for c in commits:
        files = c["code"]
        cnt_hunk = list()
        for hunk in files:
            added_hunk, removed_hunk = hunk["added"].keys(), hunk["removed"].keys()
            cnt_hunk += added_hunk + removed_hunk
        if max(cnt_hunk) <= num_hunk:
            commit_id.append(c["id"])
    return commit_id
    # cnt = [c for c in count if c <= 8]
    # print len(cnt)
    # histogram(data=count)
