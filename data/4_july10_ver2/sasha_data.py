from ultis import extract_commit_july, write_file
from sklearn.model_selection import KFold


def get_elements(commits, indexes):
    new_commits = [commits[index] for index in indexes]
    return new_commits


def creating_sasha_data(path_data_, folds, random_state):
    commits_structure = extract_commit_july(path_file=path_data_)
    commits_id = [c["id"] for c in commits_structure]
    commits_label = ["stable" if c["stable"] == "true" else "nonstable" for c in commits_structure]
    commits_id_label = [id_ + "\t" + label_ for id_, label_ in zip(commits_id, commits_label)]

    kf = KFold(n_splits=folds, random_state=random_state)
    cnt_fold = 1
    for train_index, test_index in kf.split(commits_structure):
        train_id, train_label = get_elements(commits=commits_id, indexes=train_index), get_elements(
            commits=commits_label, indexes=train_index)
        test_id, test_label = get_elements(commits=commits_id, indexes=test_index), get_elements(
            commits=commits_label, indexes=test_index)
        train_file, test_file = get_elements(commits=commits_id_label, indexes=train_index), get_elements(
            commits=commits_id_label, indexes=test_index)
        print len(train_id), len(train_label)
        print len(test_id), len(test_label)
        print len(train_file), len(test_file)

        write_file(path_file="./sasha_data/fold" + str(cnt_fold) + "/" + "train.txt", data=train_file)
        write_file(path_file="./sasha_data/fold" + str(cnt_fold) + "/" + "test.txt", data=test_file)
        cnt_fold += 1


if __name__ == "__main__":
    path_data = "./satisfy_typediff_sorted.out"
    folds_, random_state_ = 5, None  # number of folds in our dataset
    creating_sasha_data(path_data_=path_data, folds=folds_, random_state=None)
