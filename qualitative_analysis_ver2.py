from qualitative_analysis import finding_id
from ultis import write_file

if __name__ == "__main__":
    # name_root = "fold_0_1521433495_model-48550.txt"
    # path_root_ = "./statistical_test_ver2/3_mar7/"
    # path_root = path_root_ + name_root
    # print path_root
    #
    # path_label = "./data/3_mar7/typediff_test_ver2.out"
    # id_correct_root = finding_id(path_label=path_label, path_root=path_root)
    # print len(id_correct_root)
    #
    # path_baseline = "lstm_cnn_all"
    # path_baseline = path_root_ + path_baseline + ".txt"
    # id_correct_baseline = finding_id(path_label=path_label, path_root=path_baseline)
    # print len(id_correct_baseline)
    #
    # non_overlap, overlap = list(), list()
    # for i in id_correct_root:
    #     if i not in id_correct_baseline:
    #         non_overlap.append(i)
    #     else:
    #         overlap.append(i)
    #
    # print len(non_overlap), len(overlap)
    # path_write_non = "./qualitative_analysis_ver2/nonOverlap_PatchNet_LSTM+CNN.txt"
    # write_file(path_file=path_write_non, data=non_overlap)
    # path_write_over = "./qualitative_analysis_ver2/overlap_PatchNet_LSTM+CNN.txt"
    # write_file(path_file=path_write_over, data=overlap)

    #################################################################################
    #################################################################################
    # path_baseline = "msg_code_dt"
    # path_baseline = path_root_ + path_baseline + ".txt"
    # id_correct_baseline = finding_id(path_label=path_label, path_root=path_baseline)
    # print len(id_correct_baseline)

    # non_overlap, overlap = list(), list()
    # for i in id_correct_root:
    #     if i not in id_correct_baseline:
    #         non_overlap.append(i)
    #     else:
    #         overlap.append(i)
    #
    # print len(non_overlap), len(overlap)
    # path_write_non = "./qualitative_analysis_ver2/nonOverlap_PatchNet_LPU+SVM.txt"
    # write_file(path_file=path_write_non, data=non_overlap)
    # path_write_over = "./qualitative_analysis_ver2/overlap_PatchNet_LPU+SVM.txt"
    # write_file(path_file=path_write_over, data=overlap)
    #################################################################################
    #################################################################################
    name_root = "PatchNet_all"
    path_root_ = "./data/test_data_pred_results/"
    path_root = path_root_ + name_root + ".txt"
    print path_root

    path_label = "./data/test_data/merging_markus_sasha_ver1.txt"
    id_correct_root = finding_id(path_label=path_label, path_root=path_root)
    print len(id_correct_root)

    # name_baseline = "LPU_SVM_all"
    name_baseline = "lstm_cnn_code"
    path_baseline = path_root_ + name_baseline + ".txt"
    id_correct_baseline = finding_id(path_label=path_label, path_root=path_baseline)
    print len(id_correct_baseline)

    non_overlap, overlap = list(), list()
    for i in id_correct_root:
        if i not in id_correct_baseline:
            non_overlap.append(i)
        else:
            overlap.append(i)

    print len(non_overlap), len(overlap)
    path_write_non = "./qualitative_analysis_ver3/nonOverlap_" + name_root \
                     + "_" + name_baseline
    write_file(path_file=path_write_non, data=non_overlap)
    path_write_over = "./qualitative_analysis_ver3/overlap_" + name_root \
                      + "_" + name_baseline
    write_file(path_file=path_write_over, data=overlap)
