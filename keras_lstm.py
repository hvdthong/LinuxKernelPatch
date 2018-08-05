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
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPool2D, Activation, GlobalMaxPooling1D
from baselines_statistical_test import auc_score, make_dictionary, sorted_dict
from keras.layers import Input
from keras.layers import Reshape, Concatenate, Flatten
from keras.models import Model


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
    # -------------------------------------------
    # model.add(LSTM(lstm_output_size))
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

    print('Train...')
    # batch_size, num_epochs = 64, 3
    # model.fit(x_train, y_train,
    #           batch_size=FLAGS.batch_size,
    #           epochs=FLAGS.num_epochs,
    #           validation_data=(x_test, y_test))
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


def cnn_model(x_train, y_train, x_test, y_test, dictionary_size, FLAGS):
    # Convolution
    print x_train.shape, y_train.shape, x_test.shape, y_test.shape
    sequence_length = x_train.shape[1]
    vocabulary_size = dictionary_size
    embedding_dim = 8
    filter_sizes = [1, 2]
    num_filters = 8
    drop = FLAGS.dropout_keep_prob
    epochs = FLAGS.num_epochs
    batch_size = FLAGS.batch_size

    inputs = Input(shape=(sequence_length,), dtype='int32')
    embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
    reshape = Reshape((sequence_length, embedding_dim, 1))(embedding)
    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)
    maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(conv_1)
    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1])

    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=1, activation='sigmoid')(dropout)

    # this creates a model that includes
    model = Model(inputs=inputs, outputs=output)

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=FLAGS.num_epochs,
              validation_data=(x_test, y_test))
    return model


if __name__ == "__main__":
    tf = model_parameters()
    FLAGS = tf.flags.FLAGS
    print FLAGS.path
    # print_params(tf)
    exit()

    commits_ = extract_commit(path_file=FLAGS.path)
    filter_commits = filtering_commit(commits=commits_, num_file=FLAGS.code_file, num_hunk=FLAGS.code_hunk,
                                      num_loc=FLAGS.code_line,
                                      size_line=FLAGS.code_length)
    if "msg" in FLAGS.model:
        msgs_, codes_ = extract_msg(commits=filter_commits), extract_code(commits=filter_commits)
    elif "all" in FLAGS.model:
        msgs_, codes_ = extract_msg(commits=filter_commits), extract_code(commits=filter_commits)
        all_lines = add_two_list(list1=msgs_, list2=codes_)
        msgs_ = all_lines

    elif "code" in FLAGS.model:
        msgs_, codes_ = extract_msg(commits=filter_commits), extract_code(commits=filter_commits)
        msgs_ = codes_
    else:
        print "You need to type correct model"
        exit()

    dict_msg_, dict_code_ = dictionary(data=msgs_), dictionary(data=codes_)
    pad_msg = mapping_commit_msg(msgs=msgs_, max_length=FLAGS.msg_length, dict_msg=dict_msg_)
    labels = load_label_commits(commits=filter_commits)
    labels = convert_to_binary(labels)
    print pad_msg.shape, labels.shape, len(dict_msg_)
    folds = 10
    kf = KFold(n_splits=folds, random_state=FLAGS.seed)
    cntfold = 0
    timestamp = str(int(time.time()))
    accuracy, precision, recall, f1, auc = list(), list(), list(), list(), list()
    pred_dict = dict()
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
        elif FLAGS.model == "cnn_msg" or FLAGS.model == "cnn_code" or FLAGS.model == "cnn_all":
            model = cnn_model(x_train=X_train_msg, y_train=Y_train, x_test=X_test_msg,
                              y_test=Y_test, dictionary_size=len(dict_msg_), FLAGS=FLAGS)
        else:
            print "You need to give correct model name"
            exit()
        # model.save(FLAGS.model + "_" + str(cntfold) + ".h5")
        # y_pred = model.predict(X_test_msg, batch_size=FLAGS.batch_size)
        # y_pred = np.ravel(y_pred)
        # y_pred[y_pred > 0.5] = 1
        # y_pred[y_pred <= 0.5] = 0
        #
        # accuracy.append(accuracy_score(y_true=Y_test, y_pred=y_pred))
        # precision.append(precision_score(y_true=Y_test, y_pred=y_pred))
        # recall.append(recall_score(y_true=Y_test, y_pred=y_pred))
        # f1.append(f1_score(y_true=Y_test, y_pred=y_pred))
        # auc.append(auc_score(y_true=Y_test, y_pred=y_pred))

        model.save("./lstm_model_ver2/rerun_" + FLAGS.model + "_" + str(cntfold) + ".h5")
        cntfold += 1
        y_pred = model.predict(X_test_msg, batch_size=FLAGS.batch_size)
        y_pred = np.ravel(y_pred)
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
        pred_dict.update(make_dictionary(y_pred=y_pred, y_index=test_index))
        accuracy.append(accuracy_score(y_true=Y_test, y_pred=y_pred))
        precision.append(precision_score(y_true=Y_test, y_pred=y_pred))
        recall.append(recall_score(y_true=Y_test, y_pred=y_pred))
        f1.append(f1_score(y_true=Y_test, y_pred=y_pred))
        auc.append(auc_score(y_true=Y_test, y_pred=y_pred))

        # print "Accuracy of %s: %f" % (FLAGS.model, avg_list(accuracy))
        # print "Precision of %s: %f" % (FLAGS.model, avg_list(precision))
        # print "Recall of %s: %f" % (FLAGS.model, avg_list(recall))
        # print "F1 of %s: %f" % (FLAGS.model, avg_list(f1))
        # print "AUC of %s: %f" % (FLAGS.model, avg_list(auc))

        # path_file = "./statistical_test/3_mar7/" + FLAGS.model + ".txt"
        # write_file(path_file, y_pred)
        # print "Accuracy of %s: %f" % (FLAGS.model, avg_list(accuracy))
        # print "Precision of %s: %f" % (FLAGS.model, avg_list(precision))
        # print "Recall of %s: %f" % (FLAGS.model, avg_list(recall))
        # print "F1 of %s: %f" % (FLAGS.model, avg_list(f1))
        # cntfold += 1
        # exit()
    path_file = "./statistical_test_ver2/3_mar7/rerun_" + FLAGS.model + ".txt"
    write_file(path_file=path_file, data=sorted_dict(dict=pred_dict))
    print accuracy, "Accuracy and std of %s: %f %f" % (FLAGS.model, np.mean(np.array(accuracy)), np.std(np.array(accuracy)))
    print precision, "Precision of %s: %f %f" % (FLAGS.model, np.mean(np.array(precision)), np.std(np.array(precision)))
    print recall, "Recall of %s: %f %f" % (FLAGS.model, np.mean(np.array(recall)), np.std(np.array(recall)))
    print f1, "F1 of %s: %f %f" % (FLAGS.model, np.mean(np.array(f1)), np.std(np.array(f1)))
    print auc, "AUC of %s: %f %f" % (FLAGS.model, np.mean(np.array(auc)), np.std(np.array(auc)))
    print_params(tf)
    exit()
