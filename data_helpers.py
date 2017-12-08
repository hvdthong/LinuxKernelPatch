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


def padding_multiple_length(lines, max_length):
    new_lines = list()
    for l in lines:
        new_lines.append(padding_length(line=l, max_length=max_length))
    return new_lines


def padding_line(lines, max_line, max_length):
    new_lines = padding_multiple_length(lines=lines, max_length=max_length)
    if len(lines) < max_line:
        for l in xrange(0, max_line - len(lines)):
            new_lines.append(padding_length(line="", max_length=max_length))
        return new_lines
    else:
        return new_lines


def filtering_code(lines):
    new_lines = list()
    for l in lines:
        code = " ".join(l.split(":")[1].split(","))
        new_lines.append(code)
    return new_lines


def padding_hunk_code(code, max_hunk, max_line, max_length):
    new_hunks = dict()
    for i in xrange(1, max_hunk + 1):
        if i not in code.keys():
            new_hunks[i] = padding_line(lines=[""], max_line=max_line, max_length=max_length)
        else:
            new_hunks[i] = padding_line(lines=filtering_code(code[i]), max_line=max_line, max_length=max_length)
    return new_hunks


def padding_hunk(file, max_hunk, max_line, max_length):
    new_file = dict()
    new_file["removed"] = padding_hunk_code(file["removed"], max_hunk=max_hunk, max_line=max_line, max_length=max_length)
    new_file["added"] = padding_hunk_code(file["added"], max_hunk=max_hunk, max_line=max_line, max_length=max_length)
    return new_file


def padding_file(commits, max_hunk, max_line, max_length):
    # remember that we assume that we only have one file in commit code
    padding_code = list()
    for c in commits:
        files = c["code"]
        pad_file = list()
        for f in files:
            pad_file.append(padding_hunk(file=f, max_hunk=max_hunk, max_line=max_line, max_length=max_length))
        padding_code.append(pad_file)
    return padding_code


def


if __name__ == "__main__":
    path_data = "./data/oct5/sample_eq100_line_oct5.out"
    # path_data = "./data/oct5/eq100_line_oct5.out"
    commits_ = extract_commit(path_file=path_data)
    nfile, nhunk, nline, nleng = 1, 8, 10, 120
    filter_commits = filtering_commit(commits=commits_, num_file=nfile, num_hunk=nhunk, num_loc=nline, size_line=nleng)
    msgs = extract_msg(commits=filter_commits)
    codes = extract_code(commits=filter_commits)
    # print count_uniques_words(lists=msgs), count_uniques_words(lists=codes)
    # print padding_length(line="", max_length=5)
    # print padding_length(line="i love you so much", max_length=5)
    # print padding_line(lines=["", ""], max_lines=3, max_length=4)
    padding_file(commits=commits_, max_hunk=nhunk, max_line=nline, max_length=nleng)
    # print padding_hunk(file={"added": {1: ["Normal: 75,75,98"]}, "removed": {1: ["Normal: 75,75,98"]}}, max_hunk=2, max_line=2,
    #                    max_length=4)
