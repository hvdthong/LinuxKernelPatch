from os import listdir
from os.path import isfile, join


def sorted_files(files):
    root_name = files[0].split("-")[0]
    name_models = [int(file_.split("-")[-1].replace(".txt", "")) for file_ in files]
    name_models.sort()
    new_files = [root_name + "-%s.txt" % (str(n)) for n in name_models]
    # for n in name_models:
    #     new_files.append(root_name + "-%s.txt" % (str(n)))
    for n in new_files:
        print n

    print len(new_files)
    exit()
    return new_files


def check_path_files(roots, files):
    new_roots = []
    for r in roots:
        root = []
        for file_ in files:
            if r in file_:
                root.append(file_)
        new_roots.append(sorted_files(files=root))
    print len(new_roots)
    return new_roots


if __name__ == "__main__":
    # # path for PatchNet model -- commit message
    folds_path = list()
    folds_path.append("/home/jameshoang/PycharmCode/LinuxKernelPatch/data/4_july10_ver2/runs/1533725110_fold_0")
    folds_path.append("/home/jameshoang/PycharmCode/LinuxKernelPatch/data/4_july10_ver2/runs/1533725116_fold_1")
    folds_path.append("/home/jameshoang/PycharmCode/LinuxKernelPatch/data/4_july10_ver2/runs/1533725094_fold_2")
    folds_path.append("/home/jameshoang/PycharmCode/LinuxKernelPatch/data/4_july10_ver2/runs/1533725101_fold_3")
    folds_path.append("/home/jameshoang/PycharmCode/LinuxKernelPatch/data/4_july10_ver2/runs/1533725097_fold_4")
    folds_path = [f.split("/")[-1] for f in folds_path]

    path_PatchNet = "./patchNet_results"
    onlyfiles = [f for f in listdir(path_PatchNet) if isfile(join(path_PatchNet, f))]
    check_path_files(roots=folds_path, files=onlyfiles)
