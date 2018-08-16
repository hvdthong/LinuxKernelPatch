from ultis import extract_commit, filtering_commit
from baselines import extract_msg, extract_code
import numpy as np
import math


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
    elif line_length > max_length:
        line_split = line.split()
        return " ".join([line_split[i] for i in xrange(max_length)])
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
    elif len(lines) > max_line:
        return [new_lines[i] for i in xrange(max_line)]
    else:
        return new_lines


def filtering_code(lines):
    new_lines = list()
    for l in lines:
        code = " ".join(l.split(":")[1].split(","))
        code = l.split(":")[0] + " " + code
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
    new_file["removed"] = padding_hunk_code(file["removed"], max_hunk=max_hunk, max_line=max_line,
                                            max_length=max_length)
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


def dictionary(data):
    # create dictionary for commit message
    lists = list()
    for m in data:
        lists += m.split()
    lists = list(set(lists))
    lists.append("NULL")
    new_dict = dict()
    for i in xrange(len(lists)):
        new_dict[lists[i]] = i
    return new_dict


def mapping_commit_msg(msgs, max_length, dict_msg):
    pad_msg = padding_multiple_length(lines=msgs, max_length=max_length)
    new_pad_msg = list()
    for line in pad_msg:
        line_split = line.split(" ")
        new_line = list()
        for w in line_split:
            new_line.append(dict_msg[w])
        new_pad_msg.append(np.array(new_line))
    return np.array(new_pad_msg)


def mapping_commit_code_file(code, dict_code):
    new_hunks = list()
    for k in code.keys():
        hunk, new_hunk = code[k], list()
        for l in hunk:
            split_ = l.split(" ")
            new_line = list()
            for w in split_:
                new_line.append(dict_code[w])
            new_hunk.append(np.array(new_line))
        new_hunks.append(np.array(new_hunk))
    return np.array(new_hunks)


def mapping_commit_code(type, commits, max_hunk, max_code_line, max_code_length, dict_code):
    pad_code = padding_file(commits=commits, max_hunk=max_hunk, max_line=max_code_line, max_length=max_code_length)
    new_pad_code = list()
    for p in pad_code:
        file_ = p[0]  # we only use one file
        new_file = mapping_commit_code_file(code=file_[type], dict_code=dict_code)
        new_pad_code.append(new_file)
    return np.array(new_pad_code)


def load_label_commits(commits):
    # labels = [[1] if c["stable"] == "true" else [0] for c in commits]
    labels = [[0, 1] if c["stable"] == "true" else [1, 0] for c in commits]
    return np.array(labels)


def convert_to_binary(labels):
    new_labels = [1 if np.array_equal(l, np.array([0, 1])) else 0 for l in labels]
    return np.array(new_labels)


def random_mini_batch(X_msg, X_added_code, X_removed_code, Y, mini_batch_size=64, seed=0):
    m = X_msg.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X_msg = X_msg[permutation, :]
    shuffled_X_added = X_added_code[permutation, :, :, :]
    shuffled_X_removed = X_removed_code[permutation, :, :, :]
    if len(Y.shape) == 1:
        shuffled_Y = Y[permutation]
    else:
        shuffled_Y = Y[permutation, :]
    # shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / float(mini_batch_size))  # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = int(num_complete_minibatches)
    for k in range(0, num_complete_minibatches):
        mini_batch_X_msg = shuffled_X_msg[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_added = shuffled_X_added[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_X_removed = shuffled_X_removed[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        # mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X_msg, mini_batch_X_added, mini_batch_X_removed, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X_msg = shuffled_X_msg[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X_added = shuffled_X_added[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_X_removed = shuffled_X_removed[num_complete_minibatches * mini_batch_size: m, :, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        # mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X_msg, mini_batch_X_added, mini_batch_X_removed, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def mini_batches(X_msg, X_added_code, X_removed_code, Y, mini_batch_size=64, seed=0):
    m = X_msg.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: No shuffle (X, Y)
    shuffled_X_msg = X_msg
    shuffled_X_added = X_added_code
    shuffled_X_removed = X_removed_code
    shuffled_Y = Y

    # Step 2: Partition (X, Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / float(mini_batch_size))  # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = int(num_complete_minibatches)
    for k in range(0, num_complete_minibatches):
        mini_batch_X_msg = shuffled_X_msg[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_added = shuffled_X_added[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_X_removed = shuffled_X_removed[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X_msg, mini_batch_X_added, mini_batch_X_removed, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X_msg = shuffled_X_msg[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X_added = shuffled_X_added[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_X_removed = shuffled_X_removed[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X_msg, mini_batch_X_added, mini_batch_X_removed, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


if __name__ == "__main__":
    # path_data = "./data/1_oct5/sample_eq100_line_oct5.out"
    path_data = "./data/1_oct5/eq100_line_oct5.out"
    # path_data = "./data/3_mar7/typediff.out"
    commits_ = extract_commit(path_file=path_data)
    nfile, nhunk, nline, nleng = 1, 8, 10, 120
    filter_commits = filtering_commit(commits=commits_, num_file=nfile, num_hunk=nhunk, num_loc=nline, size_line=nleng)
    msgs_, codes_ = extract_msg(commits=filter_commits), extract_code(commits=filter_commits)
    dict_msg_, dict_code_ = dictionary(data=msgs_), dictionary(data=codes_)
    print "Max length of commit msg: %i" % max([len(m.split(" ")) for m in msgs_])
    print "Size of message and code dictionary: %i, %i" % (len(dict_msg_), len(dict_code_))
    pad_msg = mapping_commit_msg(msgs=msgs_, max_length=128, dict_msg=dict_msg_)
    pad_removed_code = mapping_commit_code(type="removed", commits=filter_commits, max_hunk=nhunk, max_code_line=nline,
                                           max_code_length=nleng, dict_code=dict_code_)
    pad_added_code = mapping_commit_code(type="added", commits=filter_commits, max_hunk=nhunk, max_code_line=nline,
                                         max_code_length=nleng, dict_code=dict_code_)
    print pad_msg.shape, pad_added_code.shape, pad_removed_code.shape
