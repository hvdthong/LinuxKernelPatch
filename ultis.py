from extract_commit import commits_index, commit_id, commit_stable, commit_msg, commit_date, commit_code


def load_file(path_file):
    lines = list(open(path_file, "r").readlines())
    return lines


def commit_info(commit):
    id = commit_id(commit)
    stable = commit_stable(commit)
    date = commit_date(commit)
    msg = commit_msg(commit)
    code = commit_code(commit)
    return id, stable, date, msg, code


def extract_commit(path_file):
    commits = load_file(path_file=path_file)
    indexes = commits_index(commits=commits)
    dicts = list()
    for i in xrange(0, len(indexes)):
        dict = {}
        if i == len(indexes) - 1:
            id, stable, date, msg, code = commit_info(commits[indexes[i]:])
        else:
            id, stable, date, msg, code = commit_info(commits[indexes[i]:indexes[i + 1]])
        dict["id"] = id
        dict["stable"] = stable
        dict["date"] = date
        dict["msg"] = msg
        dict["code"] = code
        dicts.append(dict)
    return dicts


def filtering_commit(num_file, num_hunk, num_loc, size_line):
    print "hello"


if __name__ == "__main__":
    # path_data = "./data/oct5/sample_eq100_line_oct5.out"
    path_data = "./data/oct5/eq100_line_oct5.out"
    commits = extract_commit(path_file=path_data)
    nfile, nhunk, nline, nleng = 1, 8, 10, 120
    print nfile, nhunk, nline, nleng
