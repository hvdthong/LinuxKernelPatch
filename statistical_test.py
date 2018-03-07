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

    print root_file
    print len(compare_files)