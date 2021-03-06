import numpy as np
from openpyxl.utils.units import points_to_pixels
from sklearn import metrics
from ultis import load_file
import matplotlib.pyplot as plt


def draw_roc_curve(path_file):
    data = load_file(path_file=path_file)
    data = np.array([float(y) for y in data])
    fpr, tpr, threshold = metrics.roc_curve(y_true, data)
    roc_auc = metrics.auc(fpr, tpr)
    return fpr, tpr, roc_auc


def draw_prc_recall_curve(y_true, path_file, point):
    data = load_file(path_file=path_file)
    data = np.array([float(y) for y in data])
    prc, rc, threshold = metrics.precision_recall_curve(y_true, data)
    new_prc, new_rc = list(), list()
    for i in xrange(0, len(prc), int(len(prc)/point)):
        new_prc.append(prc[i])
        new_rc.append(rc[i])
    return new_prc[:point], new_rc[:point]


if __name__ == "__main__":
    path_true = "./statistical_test_prob/true_label.txt"
    y_true = load_file(path_file=path_true)
    y_true = np.array([int(y) for y in y_true])

    path_sasha = "./statistical_test_prob_ver3/sasha_results.txt"
    fpr_sasha, tpr_sasha, roc_auc_sasha = draw_roc_curve(path_file=path_sasha)

    path_PatchNet = "./statistical_test_prob_ver3/PatchNet.txt"
    fpr_PatchNet, tpr_PatchNet, roc_auc_PatchNet = draw_roc_curve(path_file=path_PatchNet)

    # path_lstm = "./statistical_test_prob/lstm_cnn_all.txt"
    # fpr_lstm, tpr_lstm, roc_auc_lstm = draw_roc_curve(path_file=path_lstm)
    #
    # path_cnn = "./statistical_test_prob/cnn_all.txt"
    # fpr_cnn, tpr_cnn, roc_auc_cnn = draw_roc_curve(path_file=path_cnn)
    #
    # path_pn = "./statistical_test_prob/all_model-19825.txt"
    # fpr_pn, tpr_pn, roc_auc_pn = draw_roc_curve(path_file=path_pn)
    #
    # path_lr = "./statistical_test_prob/lr.txt"
    # fpr_lr, tpr_lr, roc_auc_lr = draw_roc_curve(path_file=path_lr)
    # print fpr_lr,    len(fpr_lr)
    # print tpr_lr, len(tpr_lr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr_PatchNet, tpr_PatchNet, 'r', label='PatchNet')  # --AUC = %0.3f' % roc_auc_lstm)
    plt.plot(fpr_sasha, tpr_sasha, 'b', label='Sasha')  # - AUC = %0.3f' % roc_auc_sasha)
    # plt.plot(fpr_cnn, tpr_cnn, 'g', label='LSTM-CNN -- AUC = %0.3f' % roc_auc_cnn)  # --AUC = %0.3f' % roc_auc_cnn)
    # # plt.plot(fpr_pn, tpr_pn, 'b--', label='Test')  # --AUC = %0.3f' % roc_auc_cnn)
    # plt.plot(fpr_lr, tpr_lr, 'g--', label='LPU-SVM -- AUC = %0.3f' % roc_auc_lr)  # --AUC = %0.3f' % roc_auc_cnn)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'b--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    # path_sasha = "./statistical_test_prob_ver3/sasha_results.txt"
    # num_point = 1000
    # pr_sasha, rc_sasha = draw_prc_recall_curve(path_file=path_sasha, point=num_point)
    # path_PatchNet = "./statistical_test_prob_ver3/PatchNet.txt"
    # pr_PatchNet, rc_PatchNet = draw_prc_recall_curve(path_file=path_PatchNet, point=num_point)
    #
    # print len(pr_sasha), len(rc_sasha)
    # print len(pr_PatchNet), len(rc_PatchNet)

    #
    # pr_sasha_select = list()
    # for i in xrange(0, len(pr_sasha), int(len(pr_sasha)/1000)):
    #     print pr_sasha[i], rc_sasha[i]
    #
    # exit()

    # plt.title('Precision Recall Curve')
    # plt.plot(pr_PatchNet, rc_PatchNet, 'b', label='PatchNet')  # - AUC = %0.3f' % roc_auc_sasha)
    # plt.plot(pr_sasha, rc_sasha, 'r', label='Sasha')  # --AUC = %0.3f' % roc_auc_lstm)
    # # plt.plot(fpr_cnn, tpr_cnn, 'g', label='LSTM-CNN -- AUC = %0.3f' % roc_auc_cnn)  # --AUC = %0.3f' % roc_auc_cnn)
    # # # plt.plot(fpr_pn, tpr_pn, 'b--', label='Test')  # --AUC = %0.3f' % roc_auc_cnn)
    # # plt.plot(fpr_lr, tpr_lr, 'g--', label='LPU-SVM -- AUC = %0.3f' % roc_auc_lr)  # --AUC = %0.3f' % roc_auc_cnn)
    # plt.legend(loc='lower left')
    # # plt.plot([0, 1], [0, 1], 'b--')
    # # plt.xlim([0, 1])
    # # plt.ylim([0, 1])
    # plt.ylabel('Precision')
    # plt.xlabel('Recall')
    # plt.show()
