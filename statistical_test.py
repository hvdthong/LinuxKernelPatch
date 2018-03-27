from ultis import load_file
from scipy import stats


def convert_to_float(data):
    new_data = [float(a) for a in data]
    return new_data


def one_tailed_test(one_file, second_file):
    one_, second_ = load_file(one_file), load_file(second_file)
    one_, second_ = convert_to_float(one_), convert_to_float(second_)
    print stats.ttest_ind(one_, second_, axis=0, equal_var=True)


if __name__ == "__main__":
    path_file = "./statistical_test/3_mar7/"
    # root_file = "./statistical_test/fold_0_1518703738_model-46656.txt"
    root_file = path_file + "fold_0_1521433495_model-48550.txt"
    compare_files = list()
        compare_files.append(path_file + "code_dt.txt")
    compare_files.append(path_file + "code_lr.txt")
    compare_files.append(path_file + "code_svm.txt")
    compare_files.append(path_file + "msg_code_dt.txt")
    compare_files.append(path_file + "msg_code_lr.txt")
    compare_files.append(path_file + "msg_code_svm.txt")
    compare_files.append(path_file + "msg_dt.txt")
    compare_files.append(path_file + "msg_lr.txt")
    compare_files.append(path_file + "msg_svm.txt")
    for f in compare_files:
        one_tailed_test(one_file=root_file, second_file=f)
