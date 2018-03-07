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
    root_file = "./statistical_test/fold_0_1518703738_model-46656.txt"
    compare_files = list()
    compare_files.append("./statistical_test/code_dt.txt")
    compare_files.append("./statistical_test/code_lr.txt")
    compare_files.append("./statistical_test/code_svm.txt")
    compare_files.append("./statistical_test/msg_code_dt.txt")
    compare_files.append("./statistical_test/msg_code_lr.txt")
    compare_files.append("./statistical_test/msg_code_svm.txt")
    compare_files.append("./statistical_test/msg_dt.txt")
    compare_files.append("./statistical_test/msg_lr.txt")
    compare_files.append("./statistical_test/msg_svm.txt")
    for f in compare_files:
        one_tailed_test(one_file=root_file, second_file=f)
