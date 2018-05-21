from ultis import extract_commit, filtering_commit, write_file


if __name__ == "__main__":
    path_data = "./data/test_data/merging_markus_sasha.txt"
    commits_ = extract_commit(path_file=path_data)
    nfile, nhunk, nline, nleng = 1, 8, 10, 120
    filter_commits = filtering_commit(commits=commits_, num_file=nfile, num_hunk=nhunk, num_loc=nline, size_line=nleng)
    ids_ = [c["id"] for c in filter_commits]
    labels_ = [c["stable"] for c in filter_commits]

    data = list()
    for i, l in zip(ids_, labels_):
        data.append(i + "\t" + l)
    path_write = "./data/test_data/merging_markus_sasha_ver1.txt"
    write_file(path_file=path_write, data=data)