from init_params import model_parameters, print_params
from ultis import load_file
from ultis import extract_commit, filtering_commit
from baselines import extract_msg, extract_code
from data_helpers import dictionary, mapping_commit_msg, mapping_commit_code, \
    load_label_commits, random_mini_batch, mini_batches, convert_to_binary
from sklearn.model_selection import KFold
from baselines import get_items
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def qualitative_analysis():
    print "hello"


tf = model_parameters()
FLAGS = tf.flags.FLAGS
print_params(tf)

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_test:
    X_test_msg, X_test_added_code, X_test_removed_code, y_test = loading_data()
else:
    print "You need to turn on the evaluating file."
    exit()