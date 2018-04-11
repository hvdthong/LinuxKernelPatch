from baselines import extract_commit, extract_msg, extract_code, add_two_list
from ultis import load_file, write_file


def create_dict(data):
    dictionary = list()
    for d in data:
        split_d = d.strip().split()
        dictionary += list(map(int, split_d))
    return list(set(dictionary))


def mapping_dict(index_, dictionary):
    new_dict = {}
    for i in index_:
        new_dict[i] = dictionary[i]
    new_list = list()
    for key, value in sorted(new_dict.iteritems()):
        new_list.append(str(key) + ": " + value)
    return new_list


if __name__ == "__main__":
    path_data = "./data/3_mar7/typediff.out"
    commits_ = extract_commit(path_file=path_data)
    msgs = extract_msg(commits=commits_)
    codes = extract_code(commits=commits_)
    all_lines = add_two_list(list1=msgs, list2=codes)
    print len(all_lines), len(commits_), len(msgs), len(codes)
    index = create_dict(all_lines)
    print len(index)

    path_dict = "./data/3_mar7/newres.dict"
    dict_index = load_file(path_file=path_dict)
    new_dict = {}
    for d in dict_index:
        split_d = d.strip().split(":")
        new_dict[int(split_d[0])] = split_d[1]
    print len(new_dict)
    new_dict = mapping_dict(index_=index, dictionary=new_dict)
    path_write = "./data/3_mar7/newres.simplified.dict"
    write_file(path_file=path_write, data=new_dict)

