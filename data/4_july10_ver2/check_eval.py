from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

if __name__ == "__main__":
    y_true = [0, 1, 0, 0]
    y_pred = [1, 1, 1, 1]

    print precision_score(y_true=y_true, y_pred=y_pred)