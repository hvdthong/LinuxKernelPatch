from ultis import extract_commit_july


if __name__ == "__main__":
    path_data = "./satisfy_typediff_sorted.out"
    commits_ = extract_commit_july(path_file=path_data)
    filter_commits = commits_
    print len(commits_), len(filter_commits)
    for c in filter_commits:
        print c["id"]
