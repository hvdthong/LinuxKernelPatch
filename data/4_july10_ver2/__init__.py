import os
import sys

split_path, goback_tokens = os.getcwd().split("/"), 2
goback_tokens = 2
path_working = "/".join(split_path[:len(split_path) - goback_tokens])
print path_working
# os.chdir(path_working + "/")
sys.path.append(path_working)