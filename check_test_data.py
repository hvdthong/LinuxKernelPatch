from ultis import load_file, extract_commit_new
from baselines import extract_msg, extract_code, add_two_list, extract_label
import numpy as np

if __name__ == "__main__":
    path_test = list()
    path_test.append("./data/test_data/markus_translated.out")
    path_test.append("./data/test_data/nicholask_translated.out")
    path_test.append("./data/test_data/sasha_translated.out")

    path_dict = "./data/3_mar7/newres.simplified.dict"
    dict_index = load_file(path_file=path_dict)
    new_dict = {}
    for d in dict_index:
        split_d = d.strip().split(":")
        new_dict[int(split_d[0])] = split_d[1]

    data = list()
    for p in path_test:
        p_data = load_file(path_file=p)
        data += p_data
    commits_ = extract_commit_new(commits=data)
    msgs = extract_msg(commits=commits_)
    codes = extract_code(commits=commits_)
    all_lines = add_two_list(list1=msgs, list2=codes)
    labels = extract_label(commits=commits_)

    # pos_label = len([1 for l in labels if l == 1])
    # neg_label = len([0 for l in labels if l == 0])
    print len(labels), np.count_nonzero(np.array(labels))

    # cnt = 1
    # for i in all_lines:
    #     split_i = i.split()
    #     for j in split_i:
    #         if int(j) == 0:
    #             print i
    #             break
    #         else:
    #             if int(j) not in new_dict.keys():
    #                 print i
    #                 break
    #             else:
    #                 print cnt
    #     cnt += 1
    # print len(all_lines)