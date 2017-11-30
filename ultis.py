from gevent.ares import result

from extract_commit import commits_index, commit_id, commit_stable, commit_msg, commit_date, commit_code
from filter_commit import filter_number_code_file, filter_number_code_hunk, filter_loc_hunk, filter_loc_len


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


def interset(lists):
    # sets = iter(map(set, d))
    # result = sets.next()
    lists = [[1, 2, 3, 4], [2, 3, 4], [3, 4, 5, 6, 7]]
    result = set(lists[0])
    for i in xrange(1, len(lists)):
        result = result.intersection(lists[i])
    return list(result)


def filtering_commit(commits, num_file, num_hunk, num_loc, size_line):
    # print len(commits)
    code_file_ids = filter_number_code_file(commits=commits, num_file=num_file)
    print len(code_file_ids)

    code_hunk_ids = filter_number_code_hunk(commits=commits, num_hunk=num_hunk)
    print len(code_hunk_ids)

    # print len(list(set(code_file_ids).intersection(code_hunk_ids)))

    loc_hunk_ids = filter_loc_hunk(commits=commits, num_loc=num_loc)
    print len(loc_hunk_ids)

    loc_len_ids = filter_loc_len(commits=commits, size_line=size_line)
    print len(loc_len_ids)

    all_ids = [code_file_ids] + [code_hunk_ids] + [loc_hunk_ids] + [loc_len_ids]
    print len(all_ids)
    print len(interset(all_ids))
    print interset(all_ids)
    exit()


if __name__ == "__main__":
    # path_data = "./data/oct5/sample_eq100_line_oct5.out"
    path_data = "./data/oct5/eq100_line_oct5.out"
    commits_ = extract_commit(path_file=path_data)
    nfile, nhunk, nline, nleng = 1, 8, 10, 120
    filtering_commit(commits=commits_, num_file=nfile, num_hunk=nhunk, num_loc=nline, size_line=nleng)
