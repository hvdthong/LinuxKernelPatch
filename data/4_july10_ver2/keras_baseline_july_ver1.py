# import os
# import sys
#
# split_path, goback_tokens = os.getcwd().split("/"), 2
# goback_tokens = 2
# path_working = "/".join(split_path[:len(split_path) - goback_tokens])
# print path_working
# # os.chdir(path_working + "/")
# sys.path.append(path_working)

from init_params import model_parameters
from keras_lstm import print_params
from ultis import extract_commit_july

tf_ = model_parameters()
FLAGS_ = tf_.flags.FLAGS
print_params(tf_)

commits_ = extract_commit_july(path_file=FLAGS_.path)
filter_commits = commits_
print len(commits_)

