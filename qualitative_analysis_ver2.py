from qualitative_analysis import finding_id

if __name__ == "__main__":
    name_root = "fold_0_1521433495_model-48550.txt"
    path_root_ = "./statistical_test_ver2/3_mar7/"
    path_root = path_root_ + name_root
    print path_root

    path_label = "./data/3_mar7/typediff_test_ver2.out"
    id_correct_root = finding_id(path_label=path_label, path_root=path_root)
    print len(id_correct_root)

    path_baseline = "lstm_cnn_all"
    path_baseline = path_root_ + path_baseline + ".txt"
    id_correct_baseline = finding_id(path_label=path_label, path_root=path_baseline)
    print len(id_correct_baseline)