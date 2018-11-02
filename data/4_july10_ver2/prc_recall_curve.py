from matplotlib.lines import lineStyles

from roc_curve_prc_rc_curve import draw_prc_recall_curve
from ggplot import *
from ultis import load_file
import numpy as np
from ggplot import meat
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == "__main__":
    path_true = "./statistical_test_prob/true_label.txt"
    y_true = load_file(path_file=path_true)
    y_true = np.array([int(y) for y in y_true])

    num_point = 1000
    path_PatchNet = "./statistical_test_prob_ver3/PatchNet.txt"
    pr_PatchNet, rc_PatchNet = draw_prc_recall_curve(y_true=y_true, path_file=path_PatchNet, point=num_point)
    # variable_PatchNet = ['PatchNet' for i in xrange(len(pr_PatchNet))]

    def average(data):
        return sum(data)/ float(len(data))

    def f1_sweetpots(pr, rc):
        new_point = []
        for p, r in zip(pr, rc):
            if p > 0.8 and r > 0.8:
                new_point.append(2 * p * r / (p + r))
        return sorted(new_point, reverse=True)

    # print average(data=pr_PatchNet), average(data=rc_PatchNet)
    print f1_sweetpots(pr=pr_PatchNet, rc=rc_PatchNet)


    path_sasha = "./statistical_test_prob_ver3/sasha_results.txt"
    pr_sasha, rc_sasha = draw_prc_recall_curve(y_true=y_true, path_file=path_sasha, point=num_point)
    # print average(data=pr_sasha), average(data=rc_sasha)
    print f1_sweetpots(pr=pr_sasha, rc=rc_sasha)
    exit()
    # variable_sasha = ['sasha' for i in xrange(len(pr_sasha))]

    path_sasha = "./statistical_test_prob_ver3/LS-CNN.txt"
    pr_lscnn, rc_lscnn = draw_prc_recall_curve(y_true=y_true, path_file=path_sasha, point=num_point)

    path_sasha = "./statistical_test_prob_ver3/LPU-SVM.txt"
    pr_lpusvm, rc_lpusvm = draw_prc_recall_curve(y_true=y_true, path_file=path_sasha, point=num_point)

    # data_pr, data_rc = pr_PatchNet + pr_sasha, rc_PatchNet + rc_sasha
    # variable_ = variable_PatchNet + variable_sasha
    # print len(data_pr), len(data_rc), len(variable_)
    #
    # data = pd.DataFrame(data={'Precision': data_pr, 'Recall': data_rc, 'algs': variable_})
    #
    # # print type(meat)
    # meat_lng = pd.melt(meat, id_vars=['date'])
    # # print meat_lng['variable']
    # # print type(meat_lng)
    # print ggplot(aes(x='date', y='value'), data=meat_lng) + geom_point() + scale_color_yhat()
    # # exit()
    # # print ggplot(data=data, aes(x='Recall', y='Precision', colour='algs')) + geom_line(stat_density)

    # Initialize the figure
    # mpl.style.use('default')
    # plt.style.use('seaborn-darkgrid')
    plt.style.use('default')
    # create a color palette
    palette = plt.get_cmap('Set1')
    # marker = '', color = palette(num),
    # plt.title('Precision Recall Curve', fontsize=20)
    # color1 = [0.3607843137254902, 0.7529411764705882, 0.3843137254901961, 0.5]
    # color2 = [0.35294117647058826, 0.6078431372549019, 0.8313725490196079, 0.5]
    # color3 = [0.9647058823529412, 0.9254901960784314, 0.33725490196078434, 0.6]
    # plt.plot(pr_PatchNet, rc_PatchNet, linestyle='-', color=color1, label='PatchNet')
    # plt.plot(pr_sasha, rc_sasha, linestyle='--', color=color2, label='F-NN')
    # plt.plot(pr_lscnn, rc_lscnn, linestyle='-.', color=color3, label='LS-CNN')
    # plt.plot(pr_lpusvm, rc_lpusvm, linestyle=':', color=palette(4), label='LPU+SVM')
    plt.plot(pr_PatchNet, rc_PatchNet, 'C1-', label='PatchNet', linewidth=2.5)
    plt.plot(pr_sasha, rc_sasha, 'C2--', label='F-NN', linewidth=2.5)
    plt.plot(pr_lscnn, rc_lscnn, 'C3-.', label='LS-CNN', linewidth=2.5)
    plt.plot(pr_lpusvm, rc_lpusvm, 'C4:', label='LPU+SVM', linewidth=2.5)
    plt.ylabel('Precision', fontsize=17)
    plt.xlabel('Recall', fontsize=17)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.gcf().subplots_adjust(bottom=0.15)
    legend = plt.legend(loc='lower left', fancybox=True, fontsize=16)
    # frame = legend.get_frame()
    # legend.get_frame().set_alpha(2.0)
    # frame.set_facecolor('white')
    # frame.set_edgecolor('black')
    # frame.set_linewidth(5)
    plt.savefig("prc_rc_curve_ver4.pdf")
    plt.show()
    # plt.figure().savefig("fig_roc_curve.pdf")
