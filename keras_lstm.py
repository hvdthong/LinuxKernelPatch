from init_params import model_parameters, print_params
from ultis import extract_commit, filtering_commit
from baselines import extract_msg, extract_code, add_two_list
from data_helpers import dictionary, mapping_commit_msg, load_label_commits
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dropout, Dense, Embedding
from data_helpers import convert_to_binary
from baselines import get_items
import numpy as np
import time
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from ultis import write_file
from baselines import avg_list
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D, Activation, GlobalMaxPooling1D


def lstm_model(x_train, y_train, x_test, y_test, dictionary_size, FLAGS):
    model = Sequential()
    model.add(Embedding(dictionary_size, FLAGS.embedding_dim_text))
    # -------------------------------------------
    # model.add(LSTM(FLAGS.hidden_dim, dropout=FLAGS.dropout_keep_prob,
    #                recurrent_dropout=FLAGS.dropout_keep_prob, activation="relu"))
    # -------------------------------------------
    # model.add(LSTM(FLAGS.hidden_dim))
    # model.add(Dropout(FLAGS.dropout_keep_prob))
    # -------------------------------------------
    model.add(LSTM(FLAGS.hidden_dim, return_sequences=True))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(FLAGS.hidden_dim, activation="relu"))
    model.add(Dropout(FLAGS.dropout_keep_prob))
    # -------------------------------------------
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    print('Train...')
    model.fit(x_train, y_train,
              batch_size=FLAGS.batch_size,
              epochs=FLAGS.num_epochs,
              validation_data=(x_test, y_test))
    return model


def bi_lstm_model(x_train, y_train, x_test, y_test, dictionary_size, FLAGS):
    model = Sequential()
    model.add(Embedding(dictionary_size, FLAGS.embedding_dim_text, input_length=FLAGS.msg_length))
    model.add(Bidirectional(LSTM(FLAGS.hidden_dim)))
    model.add(Dropout(FLAGS.dropout_keep_prob))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=FLAGS.batch_size,
              epochs=FLAGS.num_epochs,
              validation_data=(x_test, y_test))
    return model


def lstm_cnn(x_train, y_train, x_test, y_test, dictionary_size, FLAGS):
    # Convolution
    kernel_size = 5
    filters = 64
    pool_size = 4
    # LSTM
    lstm_output_size = 70

    model = Sequential()
    model.add(Embedding(dictionary_size, FLAGS.embedding_dim_text))
    model.add(Dropout(FLAGS.dropout_keep_prob))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(lstm_output_size))
    model.add(Dropout(FLAGS.dropout_keep_prob))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=FLAGS.batch_size,
              epochs=FLAGS.num_epochs,
              validation_data=(x_test, y_test))
    return model


def bi_lstm_cnn(x_train, y_train, x_test, y_test, dictionary_size, FLAGS):
    # Convolution
    kernel_size = 5
    filters = 64
    pool_size = 4
    # LSTM
    lstm_output_size = 70

    model = Sequential()
    model.add(Embedding(dictionary_size, FLAGS.embedding_dim_text))
    model.add(Dropout(FLAGS.dropout_keep_prob))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Bidirectional(LSTM(lstm_output_size)))
    model.add(Dropout(FLAGS.dropout_keep_prob))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=FLAGS.batch_size,
              epochs=FLAGS.num_epochs,
              validation_data=(x_test, y_test))
    return model


if __name__ == "__main__":
    tf = model_parameters()
    FLAGS = tf.flags.FLAGS
    print_params(tf)

    if "msg" in FLAGS.model:
        commits_ = extract_commit(path_file=FLAGS.path)
        filter_commits = filtering_commit(commits=commits_, num_file=FLAGS.code_file, num_hunk=FLAGS.code_hunk,
                                          num_loc=FLAGS.code_line,
                                          size_line=FLAGS.code_length)
        msgs_, codes_ = extract_msg(commits=filter_commits), extract_code(commits=filter_commits)
    elif "all" in FLAGS.model:
        commits_ = extract_commit(path_file=FLAGS.path)
        filter_commits = filtering_commit(commits=commits_, num_file=FLAGS.code_file, num_hunk=FLAGS.code_hunk,
                                          num_loc=FLAGS.code_line,
                                          size_line=FLAGS.code_length)
        msgs_, codes_ = extract_msg(commits=filter_commits), extract_code(commits=filter_commits)
        all_lines = add_two_list(list1=msgs_, list2=codes_)
        msgs_ = all_lines

    elif "code" in FLAGS.model:
        commits_ = extract_commit(path_file=FLAGS.path)
        filter_commits = filtering_commit(commits=commits_, num_file=FLAGS.code_file, num_hunk=FLAGS.code_hunk,
                                          num_loc=FLAGS.code_line,
                                          size_line=FLAGS.code_length)
        msgs_, codes_ = extract_msg(commits=filter_commits), extract_code(commits=filter_commits)
        msgs_ = codes_
    else:
        print "You need to type correct model"
        exit()

    dict_msg_, dict_code_ = dictionary(data=msgs_), dictionary(data=codes_)
    pad_msg = mapping_commit_msg(msgs=msgs_, max_length=FLAGS.msg_length, dict_msg=dict_msg_)
    labels = load_label_commits(commits=filter_commits)
    labels = convert_to_binary(labels)
    print pad_msg.shape, labels.shape, labels.shape, len(dict_msg_)

    kf = KFold(n_splits=FLAGS.folds, random_state=FLAGS.seed)
    cntfold = 0
    timestamp = str(int(time.time()))
    accuracy, precision, recall, f1 = list(), list(), list(), list()
    for train_index, test_index in kf.split(filter_commits):
        X_train_msg, X_test_msg = np.array(get_items(items=pad_msg, indexes=train_index)), \
                                  np.array(get_items(items=pad_msg, indexes=test_index))
        Y_train, Y_test = np.array(get_items(items=labels, indexes=train_index)), \
                          np.array(get_items(items=labels, indexes=test_index))
        if FLAGS.model == "lstm_msg" or FLAGS.model == "lstm_code" or FLAGS.model == "lstm_all":
            model = lstm_model(x_train=X_train_msg, y_train=Y_train, x_test=X_test_msg,
                               y_test=Y_test, dictionary_size=len(dict_msg_), FLAGS=FLAGS)
        elif FLAGS.model == "bi_lstm_msg" or FLAGS.model == "bi_lstm_code" or FLAGS.model == "bi_lstm_all":
            model = bi_lstm_model(x_train=X_train_msg, y_train=Y_train, x_test=X_test_msg,
                                  y_test=Y_test, dictionary_size=len(dict_msg_), FLAGS=FLAGS)
        elif FLAGS.model == "bi_lstm_cnn_msg" or FLAGS.model == "bi_lstm_cnn_code" or FLAGS.model == "bi_lstm_cnn_all":
            model = bi_lstm_cnn(x_train=X_train_msg, y_train=Y_train, x_test=X_test_msg,
                                y_test=Y_test, dictionary_size=len(dict_msg_), FLAGS=FLAGS)
        elif FLAGS.model == "lstm_cnn_msg" or FLAGS.model == "lstm_cnn_code" or FLAGS.model == "lstm_cnn_all":
            model = lstm_cnn(x_train=X_train_msg, y_train=Y_train, x_test=X_test_msg,
                                y_test=Y_test, dictionary_size=len(dict_msg_), FLAGS=FLAGS)
        else:
            print "You need to give correct model name"
            exit()
        model.save(FLAGS.model + ".h5")
        y_pred = model.predict(X_test_msg, batch_size=FLAGS.batch_size)
        y_pred = np.ravel(y_pred)
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0

        accuracy.append(accuracy_score(y_true=Y_test, y_pred=y_pred))
        precision.append(precision_score(y_true=Y_test, y_pred=y_pred))
        recall.append(recall_score(y_true=Y_test, y_pred=y_pred))
        f1.append(f1_score(y_true=Y_test, y_pred=y_pred))

        path_file = "./statistical_test/3_mar7/" + FLAGS.model + ".txt"
        write_file(path_file, y_pred)

        print "Accuracy of %s: %f" % (FLAGS.model, avg_list(accuracy))
        print "Precision of %s: %f" % (FLAGS.model, avg_list(precision))
        print "Recall of %s: %f" % (FLAGS.model, avg_list(recall))
        print "F1 of %s: %f" % (FLAGS.model, avg_list(f1))
        print_params(tf)
        exit()
