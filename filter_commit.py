def filter_number_code_file(commits, numfile):
    # check the number of code file in the commit code
    commit_id = list()
    for c in commits:
        if len(c["code"]) <= numfile:
            commit_id.append(c["id"])
    return commit_id