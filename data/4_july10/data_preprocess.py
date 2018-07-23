from extract_commit import commits_index, commit_id, commit_date_july
from ultis import load_file, write_file
import operator


def get_commit_id_and_date(path_data_):
    commits = load_file(path_data_)
    indexes = commits_index(commits=commits)
    dicts = {}
    for i in xrange(0, len(indexes)):
        if i == len(indexes) - 1:
            date = commit_date_july(commit=commits[indexes[i]:])
        else:
            date = commit_date_july(commit=commits[indexes[i]:indexes[i + 1]])
        dicts[i] = int(date)
    sort_dicts = sorted(dicts.items(), key=operator.itemgetter(1))
    new_commits = list()
    for d in sort_dicts:
        index, date = d[0], d[1]
        print index, date
        if index == len(sort_dicts) - 1:
            new_commits += commits[indexes[index]:]
        else:
            new_commits += commits[indexes[index]:indexes[index + 1]]
    write_file("./typediff_sorted.out", new_commits)
    # new_dicts = {}
    # for i in xrange(0, len(indexes)):
    #     new_dicts[i] = sort_dicts[i][1]
    #
    # for i in xrange(0, len(indexes)):
    #     print new_dicts[i]
    # print len(new_dicts)

    # for i in sort_dicts:
    #     # print sort_dicts.keys(i), sort_dicts.values(i)
    #     print i, type(i), i[0], i[1]
    # print type(sort_dicts)
    # for i, j in enumerate(dicts):
    #     print i, dicts[i]
    exit()
    exit()
    # date_commits = commit_date_july(commit=commits)
    # id_commits = commit_id(commit=commits)
    #
    # print len(index_commits), len(date_commits), len(id_commits)
    # print date_commits[0], id_commits[0], index_commits[0]


if __name__ == "__main__":
    path_data = "typediff.out"
    get_commit_id_and_date(path_data_=path_data)
