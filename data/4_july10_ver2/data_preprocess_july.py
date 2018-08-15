from extract_commit import commits_index, commit_id, commit_date_july
from ultis import load_file, write_file
import operator
from ultis import extract_commit_july, filtering_commit_union


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
    # write_file("./typediff_sorted.out", new_commits)
    write_file(path_data_ + ".sorted", new_commits)


def get_commit_satisfy_condition(path_data_, nfile, nhunk, nline, nleng):
    commits_structure = extract_commit_july(path_file=path_data_)
    # nfile, nhunk, nline, nleng = 1, 8, 10, 120
    filter_commits = filtering_commit_union(commits=commits_structure, num_file=nfile, num_hunk=nhunk, num_loc=nline,
                                            size_line=nleng)
    print len(commits_structure), len(filter_commits)

    commits = load_file(path_data_)
    indexes = commits_index(commits=commits)
    new_commits = list()
    for i in xrange(0, len(indexes)):
        if i == len(indexes) - 1:
            id = commit_id(commit=commits[indexes[i]:])
            if id in filter_commits:
                new_commits += commits[indexes[i]:]
        else:
            id = commit_id(commit=commits[indexes[i]:indexes[i + 1]])
            if id in filter_commits:
                new_commits += commits[indexes[i]:indexes[i + 1]]
        print i, id
    # write_file("./satisfy_typediff_sorted.out", new_commits)
    write_file(path_data_ + ".satisfy", new_commits)


if __name__ == "__main__":
    # sorted information based on commit date
    # path_data = "./typediff.out"
    # get_commit_id_and_date(path_data_=path_data)

    # path_data = "./newres_funcalls_jul28.out"
    # get_commit_id_and_date(path_data_=path_data)

    # copy valid commits
    # path_data = "./typediff_sorted.out"
    # nfile_, nhunk_, nline_, nleng_ = 1, 8, 10, 120
    # get_commit_satisfy_condition(path_data_=path_data, nfile=nfile_, nhunk=nhunk_, nline=nline_, nleng=nleng_)

    path_data = "./newres_funcalls_jul28.out.sorted"
    nfile_, nhunk_, nline_, nleng_ = 1, 8, 10, 120
    get_commit_satisfy_condition(path_data_=path_data, nfile=nfile_, nhunk=nhunk_, nline=nline_, nleng=nleng_)

    # commits_ = extract_commit_july(path_file=path_data)
    # nfile, nhunk, nline, nleng = 1, 8, 10, 120
    # filter_commits = filtering_commit_union(commits=commits_, num_file=nfile, num_hunk=nhunk, num_loc=nline,
    #                                         size_line=nleng)
    # print len(commits_), len(filter_commits)
    # for f in filter_commits:
    #     print f
