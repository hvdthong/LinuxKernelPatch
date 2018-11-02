from ultis import extract_commit_july
from qualitative_analysis import load_probability_score
# from matplotlib import pyplot as plt
# import numpy as np
# from matplotlib_venn import venn3, venn3_circles, venn2, venn2_unweighted
# from matplotlib_venn._venn3 import compute_venn3_subsets
# from matplotlib_venn._venn2 import compute_venn2_subsets
import matplotlib
# matplotlib.use('Agg')
import venn


def find_stable_id(ids, true, predict):
    stables = list()
    for i, t, p in zip(ids, true, predict):
        if t == p and t == 1:
            stables.append(i)
    return set(stables)


if __name__ == '__main__':
    path_data = "./newres_funcalls_jul28.out.sorted.satisfy"
    commits_ = extract_commit_july(path_file=path_data)
    commits_id = [c["id"] for c in commits_]
    print len(commits_), len(commits_id)

    true_label = load_probability_score(model="true_label", threshold=None)
    patchNet = load_probability_score(model="PatchNet", threshold=None)
    # sasha = load_probability_score(model="sasha_results", threshold=50)
    # lstm_cnn = load_probability_score(model="LS-CNN", threshold=None)
    lpu_svm = load_probability_score(model="LPU-SVM", threshold=None)
    #
    # print len(patchNet), len(lstm_cnn), len(sasha)
    # stable_patchNet = find_stable_id(ids=commits_id, true=true_label, predict=patchNet)
    # stable_sasha = find_stable_id(ids=commits_id, true=true_label, predict=sasha)
    # stable_lstm_cnn = find_stable_id(ids=commits_id, true=true_label, predict=lstm_cnn)
    # print len(stable_patchNet), len(stable_sasha), len(stable_lstm_cnn)

    # # v = venn3(subsets=(10, 8, 22, 6, 9, 4, 0), set_labels=('Group A', 'Group B', 'Group C'))
    # # # c = venn3_circles(subsets=(10, 8, 22, 6, 9, 4, 2), linestyle='dashed', linewidth=1, color="grey")
    # # plt.show()
    #
    # # set1 = set(['A', 'B', 'C', 'D'])
    # # set2 = set(['B', 'C', 'D', 'E'])
    # # set3 = set(['C', 'D', ' E', 'F', 'G'])
    # # print compute_venn3_subsets(set1, set2, set3)
    # # print compute_venn3_subsets(set1, set2, set3)[3]
    # # print len(compute_venn3_subsets(set1, set2, set3))
    # print compute_venn3_subsets(stable_patchNet, stable_sasha, stable_patchNet)
    # # exit()
    # # venn3([set1, set2, set3], ('Set1', 'Set2', 'Set3'))
    # # plt.show()
    #
    # # Custom text labels: change the label of group A
    # v = venn3(subsets=(1746, 3218, 556, 746, 8210, 974, 31158), set_labels=('Group A', 'Group B', 'Group C'))
    # # v.get_label_by_id('A').set_text('My Favourite group!')
    # plt.show()
    
    # print compute_venn2_subsets(stable_patchNet, stable_sasha)

    # # plt.style.use('seaborn-darkgrid')
    # v = venn2_unweighted(subsets=(7326, 1982, 31138), set_labels=('PatchNet', 'F-NN'))
    # # # v.get_label_by_id('A').set_text('My Favourite group!')
    # plt.show()

    # labels = {'01': '1,982', '10': '7,326', '11': '31,138'}
    # fig, ax = venn.venn2(labels, names=['PatchNet', 'F-NN'])
    # fig.savefig('venn_patchNet&F-NN.pdf', bbox_inches='tight')
    # fig.show()

    # labels = venn.get_labels([range(10), range(5, 15), range(3, 8)], fill=['number', 'logic'])
    # labels = {'010': '1,708', '011': '274', '001': '1,848', '111': '30,006',
    #           '110': '1,132', '100': '6,164', '101': '1,162'}
    # print labels
    # fig, ax = venn.venn3(labels, names=['PatchNet', 'F-NN', 'LS-CNN'])
    # fig.savefig('ven_digram.pdf', bbox_inches='tight')
    # fig.show()

    labels = venn.get_labels([range(10), range(5, 15), range(3, 8), range(8, 17), range(10, 20)],
                             fill=['number', 'logic'])
    print labels
    exit()
    fig, ax = venn.venn5(labels, names=['list 1', 'list 2', 'list 3', 'list 4', 'list 5'])
    fig.savefig('venn7.png', bbox_inches='tight')
    fig.show()
