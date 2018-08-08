import os
from ultis import load_file, extract_commit_july
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from baselines_statistical_test import auc_score
import numpy as np
from train_PatchNet import split_train_test
from baselines import get_items
import random


def check_fold(string):
    fold_string = string.split("/")[-1]
    fold_string = fold_string.replace("fold", "")
    fold_string = fold_string.replace(".txt", "")
    return int(fold_string)


def checking_performance(id_gt, label_gt, patches):
    prob_patches = patches[0][1]
    new_labels, pred_labels = list(), list()

    new_test_dl_labels = list()
    for k, v in patches:
        if v / prob_patches >= 0.5:
            pred_labels.append(1)
        else:
            pred_labels.append(0)
        index = id_gt.index(k)
        true_label = label_gt[index]
        new_test_dl_labels.append(test_dl_labels[index])
        if true_label == "true":
            new_labels.append(1)
        else:
            new_labels.append(0)
    print len(new_labels), len(pred_labels), len(new_test_dl_labels)
    true_list, false_list = [], []
    for l in new_labels:
        if l == 1:
            true_list.append(l)
        else:
            false_list.append(l)
    print len(true_list), len(false_list)
    acc = accuracy_score(y_true=new_labels, y_pred=pred_labels)
    prc = precision_score(y_true=new_labels, y_pred=pred_labels)
    rc = recall_score(y_true=new_labels, y_pred=pred_labels)
    f1 = f1_score(y_true=new_labels, y_pred=pred_labels)
    auc = auc_score(y_true=new_labels, y_pred=pred_labels)
    print acc, prc, rc, f1, auc

    dl_acc = accuracy_score(y_true=new_labels, y_pred=new_test_dl_labels)
    dl_prc = precision_score(y_true=new_labels, y_pred=new_test_dl_labels)
    dl_rc = recall_score(y_true=new_labels, y_pred=new_test_dl_labels)
    dl_f1 = f1_score(y_true=new_labels, y_pred=new_test_dl_labels)
    dl_auc = auc_score(y_true=new_labels, y_pred=new_test_dl_labels)
    print dl_acc, dl_prc, dl_rc, dl_f1, dl_auc

    return acc, prc, rc, f1, auc


def checking_performance_v2(id_gt, label_gt, patches):
    new_labels, pred_labels = list(), list()
    new_patches_id, new_patches_label = list(), list()
    for p in patches:
        id_, label_ = p[0], p[1]
        new_patches_id.append(id_)
        new_patches_label.append(label_)
    prob_patches = patches[0][1]
    top_patches = int(len(patches) * 0.42)
    prob_patches = new_patches_label[top_patches - 1]

    # print prob_patches
    # exit()
    # print top_patches
    # exit()
    for id_, true_label in zip(id_gt, label_gt):
        if true_label == "true":
            new_labels.append(1)
        else:
            new_labels.append(0)
        if id_ in new_patches_id:
            index_ = new_patches_id.index(id_)
            # if new_patches_label[index_]/prob_patches >= 0.5:
            # if index_ <= top_patches:
            if new_patches_label[index_] > prob_patches:
                pred_labels.append(1)
            else:
                pred_labels.append(0)
        else:
            # pred_labels.append(random.choice((1, 10)))
            pred_labels.append(0)

    print len(new_labels), len(pred_labels)
    true_list, false_list = [], []
    for l in new_labels:
        if l == 1:
            true_list.append(l)
        else:
            false_list.append(l)
    print len(true_list), len(false_list)

    acc = accuracy_score(y_true=new_labels, y_pred=pred_labels)
    prc = precision_score(y_true=new_labels, y_pred=pred_labels)
    rc = recall_score(y_true=new_labels, y_pred=pred_labels)
    f1 = f1_score(y_true=new_labels, y_pred=pred_labels)
    auc = auc_score(y_true=new_labels, y_pred=pred_labels)
    print acc, prc, rc, f1, auc

    # dl_acc = accuracy_score(y_true=new_labels, y_pred=test_dl_labels)
    # dl_prc = precision_score(y_true=new_labels, y_pred=test_dl_labels)
    # dl_rc = recall_score(y_true=new_labels, y_pred=test_dl_labels)
    # dl_f1 = f1_score(y_true=new_labels, y_pred=test_dl_labels)
    # dl_auc = auc_score(y_true=new_labels, y_pred=test_dl_labels)
    # print dl_acc, dl_prc, dl_rc, dl_f1, dl_auc

    return acc, prc, rc, f1, auc


def load_results(id_gt, label_gt, single_file):
    lines = load_file(single_file)
    patches = dict()
    for l in lines:
        # patch = dict()
        split_l = l.split()
        # patch["id"], patch["score"] = split_l[0], float(split_l[1])
        # patches.append(patch)
        patches[split_l[0]] = float(split_l[1])

    patches = sorted(patches.items(), key=lambda x: x[1], reverse=True)
    # acc, prc, rc, f1, auc = checking_performance(id_gt=id_gt, label_gt=label_gt, patches=patches)
    acc, prc, rc, f1, auc = checking_performance_v2(id_gt=id_gt, label_gt=label_gt, patches=patches)
    return acc, prc, rc, f1, auc


if __name__ == "__main__":
    root_path_ = "./sasha_results/"
    files_path = ([root_path_ + x for x in os.listdir(root_path_)])
    print files_path

    root_gt = "./satisfy_typediff_sorted.out"
    commits_ = extract_commit_july(path_file=root_gt)
    commits_id_ = [c["id"] for c in commits_]
    commits_label_ = [c["stable"] for c in commits_]

    dl_labels = load_file("./statistical_test/lstm_cnn_all.txt")
    dl_labels = [float(l) for l in dl_labels]

    fold_index = split_train_test(data=commits_, folds=5, random_state=None)

    accuracy_, precision_, recall_, f1_, auc_ = list(), list(), list(), list(), list()
    for f in files_path:
        print f
        test_fold = fold_index[check_fold(string=f) - 1]["test"]
        test_commit_id, test_commit_label = get_items(items=commits_id_, indexes=test_fold), get_items(
            items=commits_label_, indexes=test_fold)
        test_dl_labels = get_items(items=dl_labels, indexes=test_fold)
        print len(test_commit_id), len(test_commit_label), len(test_dl_labels)

        acc, prc, rc, f1, auc = load_results(id_gt=test_commit_id, label_gt=test_commit_label, single_file=f)
        accuracy_.append(acc)
        precision_.append(prc)
        recall_.append(rc)
        f1_.append(f1)
        auc_.append(auc)
        # print acc, prc, rc, f1, auc
        # break

    print accuracy_, "Accuracy and std: %f %f" % (np.mean(np.array(accuracy_)), np.std(np.array(accuracy_)))
    print precision_, "Precision: %f %f" % (np.mean(np.array(precision_)), np.std(np.array(precision_)))
    print recall_, "Recall: %f %f" % (np.mean(np.array(recall_)), np.std(np.array(recall_)))
    print f1_, "F1: %f %f" % (np.mean(np.array(f1_)), np.std(np.array(f1_)))
    print auc_, "AUC: %f %f" % (np.mean(np.array(auc_)), np.std(np.array(auc_)))
