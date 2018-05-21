from ultis import extract_commit, filtering_commit, load_file


def filter_number_code_hunk(commits):
    commit_id = list()
    for c in commits:
        files = c["code"]
        cnt_hunk = list()
        for hunk in files:
            added_hunk, removed_hunk = hunk["added"].keys(), hunk["removed"].keys()
            cnt_hunk += added_hunk + removed_hunk
        # if max(cnt_hunk) <= num_hunk:
        #     commit_id.append(c["id"])
        print c["id"], max(cnt_hunk)


if __name__ == "__main__":
    path_data = "./data/test_data/merging_markus_sasha.txt"
    commits_ = extract_commit(path_file=path_data)
    nfile, nhunk, nline, nleng = 1, 8, 10, 120
    filter_commits = filtering_commit(commits=commits_, num_file=nfile, num_hunk=nhunk, num_loc=nline, size_line=nleng)
    ids_ = [c["id"] for c in filter_commits]
    labels_ = [c["stable"] for c in filter_commits]

    path_nonoverlap = "./qualitative_analysis_ver3/nonOverlap_PatchNet_all_LPU_SVM_all"
    id_overlap = load_file(path_file=path_nonoverlap)
    new_commits = list()
    for i in id_overlap:
        index_i = ids_.index(i)
        new_commits.append(filter_commits[index_i])
    filter_number_code_hunk(commits=new_commits)

