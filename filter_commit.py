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


def filter_loc_hunk(commits, num_loc):
    count = list()
    for c in commits:
        files = c["code"]
        cnt_loc_hunk = list()
        for hunk in files:
            first = hunk
            # max_hunk = max(first["removed"].keys() + first["added"].keys())
            # if max_hunk <= 8:
            removed_code, added_code = hunk["removed"], hunk["added"]
            if len(hunk["removed"].keys()) > 0:
                max_removed_code = max([len(removed_code[k]) for k in hunk["removed"].keys()])
            else:
                max_removed_code = 0
            if len(first["added"].keys()) > 0:
                max_added_code = max([len(added_code[k]) for k in first["added"].keys()])
            else:
                max_added_code = 0
            # max_added_code = max([len(added_code[k]) for k in first["added"].keys()])
            count.append(max(max_removed_code, max_added_code))
