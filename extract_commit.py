def commits_index(commits):
    commits_index = [i for i, c in enumerate(commits) if c.startswith("commit:")]
    return commits_index


def commit_id(commit):
    id = commit[0].strip().split(":")[1].strip()
    return id


def commit_stable(commit):
    stable = commit[1].strip().split(":")[1].strip()
    return stable


def commit_date(commit):
    committer_date = commit[4].strip().split(":")[1].strip()
    return committer_date


def commit_msg(commit):
    commit_msg = commit[11].strip()
    return commit_msg


def extract_hunk_code(code, sign):
    dict_hunk = {}
    for l in code:
        if sign in l:
            hunk_idx = int(l.strip().split(":")[0])
            line = l.strip().split(":")[3].strip()
            prop_line = l.strip().split(":")[2].strip()
            new_line = prop_line + ":" + line
            if hunk_idx not in dict_hunk.keys():
                dict_hunk[hunk_idx] = [new_line]
            else:
                dict_hunk[hunk_idx].append(new_line)
    return dict_hunk


def hunk_code(code):
    added_code = extract_hunk_code(code=code, sign="+")
    removed_code = extract_hunk_code(code=code, sign="-")
    return added_code, removed_code


def commit_code(commit):
    all_code = commit[14:]
    file_index = [i for i, c in enumerate(all_code) if c.startswith("file:")]
    dicts = list()
    for i in xrange(0, len(file_index)):
        dict_code = {}
        if i == len(file_index) - 1:
            added_code, removed_code = hunk_code(all_code[file_index[i]:])
        else:
            added_code, removed_code = hunk_code(all_code[file_index[i]:file_index[i + 1]])
        dict_code[i] = all_code[file_index[i]].split(":")[1].strip()
        dict_code["added"] = added_code
        dict_code["removed"] = removed_code
        dicts.append(dict_code)
    return dicts