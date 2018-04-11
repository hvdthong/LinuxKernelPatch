from baselines import extract_commit, extract_msg, extract_code, add_two_list


def create_dict(data):
    dictionary = list()
    for d in data:
        split_d = d.strip().split()
        dictionary += split_d
    return set(dictionary)


if __name__ == "__main__":
    path_data = "./data/3_mar7/typediff.out"
    commits_ = extract_commit(path_file=path_data)
    msgs = extract_msg(commits=commits_)
    codes = extract_code(commits=commits_)
    all_lines = add_two_list(list1=msgs, list2=codes)
    print len(all_lines), len(commits_), len(msgs), len(codes)
    dictionary = create_dict(all_lines)
    print len(dictionary)

    path_dict = "./data/3_mar7/new_res.dict"
