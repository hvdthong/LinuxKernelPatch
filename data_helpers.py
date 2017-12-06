from ultis import extract_commit, filtering_commit
from baselines import extract_msg, extract_code


def count_uniques_words(lists):
    lines = list()
    for l in lists:
        lines += l.split()
    return len(set(lines))


def padding_length(line, max_length):
    line_length = len(line.split())
    if line_length < max_length:
        new_line = line + " NULL" * (max_length - line_length)
        return new_line.strip()
    else:
        return line

if __name__ == "__main__":
    path_data = "./data/oct5/sample_eq100_line_oct5.out"
    # path_data = "./data/oct5/eq100_line_oct5.out"
    commits_ = extract_commit(path_file=path_data)
    nfile, nhunk, nline, nleng = 1, 8, 10, 120
    filter_commits = filtering_commit(commits=commits_, num_file=nfile, num_hunk=nhunk, num_loc=nline, size_line=nleng)
    msgs = extract_msg(commits=filter_commits)
    codes = extract_code(commits=filter_commits)
    print count_uniques_words(lists=msgs), count_uniques_words(lists=codes)
    print padding_length(line="i love you", max_length=5)
    print padding_length(line="i love you so much", max_length=5)
