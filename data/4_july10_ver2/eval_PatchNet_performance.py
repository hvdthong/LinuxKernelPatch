import os
import sys

split_path, goback_tokens = os.getcwd().split("/"), 2
goback_tokens = 2
path_working = "/".join(split_path[:len(split_path) - goback_tokens])
print path_working
# os.chdir(path_working + "/")
sys.path.append(path_working)

from os import listdir
from os.path import isfile, join
from ultis import load_file, write_file


def sorted_files(files):
    root_name = files[0].split("-")[0]
    name_models = [int(file_.split("-")[-1].replace(".txt", "")) for file_ in files]
    name_models.sort()
    new_files = [root_name + "-%s.txt" % (str(n)) for n in name_models]
    return new_files


def restruct_root(roots, path, type):
    min_files = min([len(r) for r in roots])
    new_roots = [r[: min_files] for r in roots]
    print len(new_roots), len(new_roots[0])

    for i in xrange(0, len(new_roots[0])):
        model = list()
        for j in xrange(0, len(new_roots)):
            print path + "/" + new_roots[j][i]
            model += load_file(path + "/" + new_roots[j][i])
        model_name = "model-" + new_roots[j][i].split("-")[-1].replace(".txt", "")
        print type, model_name
        # exit()
        path_write = "./patchNet_mergeResults/%s_%s.txt" % (type, model_name)
        write_file(path_file=path_write, data=model)
    return None


def check_path_files(roots, files, path, type):
    new_roots = []
    for r in roots:
        root = []
        for file_ in files:
            if r in file_:
                root.append(file_)
        new_roots.append(sorted_files(files=root))
    print len(new_roots)
    restruct_root(roots=new_roots, path=path, type=type)
    return new_roots


if __name__ == "__main__":
    # # # path for PatchNet model -- commit message
    # folds_path = list()
    # folds_path.append("/home/jameshoang/PycharmCode/LinuxKernelPatch/data/4_july10_ver2/runs/1533725110_fold_0")
    # folds_path.append("/home/jameshoang/PycharmCode/LinuxKernelPatch/data/4_july10_ver2/runs/1533725116_fold_1")
    # folds_path.append("/home/jameshoang/PycharmCode/LinuxKernelPatch/data/4_july10_ver2/runs/1533725094_fold_2")
    # folds_path.append("/home/jameshoang/PycharmCode/LinuxKernelPatch/data/4_july10_ver2/runs/1533725101_fold_3")
    # folds_path.append("/home/jameshoang/PycharmCode/LinuxKernelPatch/data/4_july10_ver2/runs/1533725097_fold_4")
    # folds_path = [f.split("/")[-1] for f in folds_path]
    # type_data = "msg"

    # # # path for PatchNet model -- commit message
    # folds_path = list()
    # folds_path.append("/home/jameshoang/PycharmCode/LinuxKernelPatch/data/4_july10_ver2/runs/1533468623_fold_0")
    # folds_path.append("/home/jameshoang/PycharmCode/LinuxKernelPatch/data/4_july10_ver2/runs/1533468722_fold_1")
    # folds_path.append("/home/jameshoang/PycharmCode/LinuxKernelPatch/data/4_july10_ver2/runs/1533470414_fold_2")
    # folds_path.append("/home/jameshoang/PycharmCode/LinuxKernelPatch/data/4_july10_ver2/runs/1533470473_fold_3")
    # folds_path.append("/home/jameshoang/PycharmCode/LinuxKernelPatch/data/4_july10_ver2/runs/1533470539_fold_4")
    # folds_path = [f.split("/")[-1] for f in folds_path]
    # type_data = "all"

    # # path for PatchNet model -- commit message
    folds_path = list()
    folds_path.append("/home/jameshoang/PycharmCode/LinuxKernelPatch/data/4_july10_ver2/runs/1533728035_fold_0")
    folds_path.append("/home/jameshoang/PycharmCode/LinuxKernelPatch/data/4_july10_ver2/runs/1533728029_fold_1")
    folds_path.append("/home/jameshoang/PycharmCode/LinuxKernelPatch/data/4_july10_ver2/runs/1533728051_fold_2")
    folds_path.append("/home/jameshoang/PycharmCode/LinuxKernelPatch/data/4_july10_ver2/runs/1533728044_fold_3")
    folds_path.append("/home/jameshoang/PycharmCode/LinuxKernelPatch/data/4_july10_ver2/runs/1533728078_fold_4")
    folds_path = [f.split("/")[-1] for f in folds_path]
    type_data = "code"

    path_PatchNet_ = "./patchNet_results_ver2"
    onlyfiles = [f for f in listdir(path_PatchNet_) if isfile(join(path_PatchNet_, f))]
    check_path_files(roots=folds_path, files=onlyfiles, path=path_PatchNet_, type=type_data)
