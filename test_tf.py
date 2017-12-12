import tensorflow as tf
import numpy as np
from data_helpers import extract_commit, filtering_commit, extract_msg, extract_code, dictionary, mapping_commit_msg, mapping_commit_code

# indices = tf.placeholder(tf.int32, [3])
# depth = tf.placeholder(tf.int32)
# one_hot = tf.one_hot(indices=indices, depth=depth)
# init = tf.initialize_all_variables()
# with tf.Session() as sess:
#     sess.run(init)
#     print sess.run(one_hot, feed_dict={indices: np.array([0, 1, 2]), depth: 4}).shape
#     print sess.run(one_hot, feed_dict={indices: np.array([0, 1, 2]), depth: 4})
# sess.close()

if __name__ == "__main__":
    path_data = "./data/oct5/sample_eq100_line_oct5.out"
    # path_data = "./data/oct5/eq100_line_oct5.out"
    commits_ = extract_commit(path_file=path_data)
    nfile, nhunk, nline, nleng = 1, 8, 10, 120
    filter_commits = filtering_commit(commits=commits_, num_file=nfile, num_hunk=nhunk, num_loc=nline, size_line=nleng)
    msgs_, codes_ = extract_msg(commits=filter_commits), extract_code(commits=filter_commits)
    dict_msg_, dict_code_ = dictionary(data=msgs_), dictionary(data=codes_)
    print "Max length of commit msg: %i" % max([len(m.split(" ")) for m in msgs_])
    print "Size of message and code dictionary: %i, %i" % (len(dict_msg_), len(dict_code_))
    pad_msg = mapping_commit_msg(msgs=msgs_, max_length=128, dict_msg=dict_msg_)
    pad_code = mapping_commit_code(commits=filter_commits, max_hunk=nhunk, max_code_line=nline, max_code_length=nleng, dict_code=dict_code_)
    print pad_msg.shape, pad_code.shape

    matrix = np.random.rand(1, 2, 2, 2, 3)
    print matrix
    print "--------------------------"
    print np.mean(matrix, axis=3)
    print np.mean(matrix, axis=3).shape
