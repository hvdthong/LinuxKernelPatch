from ultis import extract_commit
import matplotlib.pyplot as plt


def histogram(data):
    plt.hist(data)  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()


def check_number_code_file(commits):
    # check the number of code file in the commit code
    count = list()
    for c in commits:
        count.append(len(c["code"]))
    cnt_2 = [c for c in count if c == 2]
    cnt_1 = [c for c in count if c == 1]
    print len(cnt_2), len(cnt_1)
    histogram(data=count)


def check_number_hunk(commits):
    count = list()
    for c in commits:
        code = c["code"]
        if len(code) == 1:
            first = code[0]
            count.append(max(first["removed"].keys() + first["added"].keys()))
    cnt = [c for c in count if c <= 8]
    print len(cnt)
    histogram(data=count)


def check_number_loc_in_hunk(commits):
    count = list()
    for c in commits:
        code = c["code"]
        if len(code) == 1:
            first = code[0]
            max_hunk = max(first["removed"].keys() + first["added"].keys())
            if max_hunk <= 8:
                removed_code, added_code = first["removed"], first["added"]
                if len(first["removed"].keys()) > 0:
                    max_removed_code = max([len(removed_code[k]) for k in first["removed"].keys()])
                else:
                    max_removed_code = 0
                if len(first["added"].keys()) > 0:
                    max_added_code = max([len(added_code[k]) for k in first["added"].keys()])
                else:
                    max_added_code = 0
                # max_added_code = max([len(added_code[k]) for k in first["added"].keys()])
                count.append(max(max_removed_code, max_added_code))
    cnt = [c for c in count if c <= 10]
    print len(cnt)
    histogram(data=cnt)


def check_number_lengcode_in_hunk(commits):
    count = list()
    for c in commits:
        code = c["code"]
        if len(code) == 1:
            first = code[0]
            max_hunk = max(first["removed"].keys() + first["added"].keys())
            if max_hunk <= 8:
                removed_code, added_code = first["removed"], first["added"]
                if len(first["removed"].keys()) > 0:
                    max_removed_code = max([len(removed_code[k]) for k in first["removed"].keys()])
                else:
                    max_removed_code = 0
                if len(first["added"].keys()) > 0:
                    max_added_code = max([len(added_code[k]) for k in first["added"].keys()])
                else:
                    max_added_code = 0
                # max_added_code = max([len(added_code[k]) for k in first["added"].keys()])
                if max(max_removed_code, max_added_code) <= 10:
                    line = list()
                    for k in first["removed"].keys():
                        for l in removed_code[k]:
                            line.append(len(l.split(",")))
                    for k in first["added"].keys():
                        for l in added_code[k]:
                            line.append(len(l.split(",")))
                    count.append(max(line))
    cnt = [c for c in count if c <= 120]
    print len(cnt), len(count)
    histogram(data=cnt)


if __name__ == "__main__":
    path_data = "./data/oct5/eq100_line_oct5.out"
    commits_ = extract_commit(path_file=path_data)
    # check_number_code_file(commits=commits_)  # we choose commits which have number of code file  <rcal= 1
    # check_number_hunk(commits=commits_)  # we choose commits which have number of hunk <= 8
    # check_number_loc_in_hunk(commits=commits_)  # we choose commits which have number of lines of code <= 10
    check_number_lengcode_in_hunk(commits=commits_)  # we choose commits which have number of lines of code <= 120
