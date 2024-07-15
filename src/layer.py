from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import numpy as np
import time

class Layer:
    def __init__(self, num_forests, n_estimators, num_classes,
                 n_fold, layer_index, max_depth=100, min_samples_leaf=1):
        self.num_forests = num_forests  # number of forests
        self.n_estimators = n_estimators  # number of trees in each forest
        self.num_classes = num_classes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_fold = n_fold
        self.layer_index = layer_index
        self.forest_list = []

    def train(self, train_data, train_label):
        val_prob = np.zeros((self.num_forests, train_data.shape[0], self.num_classes), dtype=np.float64)
        n_dim = train_data.shape[1]

        for forest_index in range(self.num_forests):
            num_classes = int(np.max(train_label)+1)
            val_prob_forest = np.zeros((train_data.shape[0], num_classes))

            tempk = KFold(n_splits=self.n_fold, shuffle=True)
            kf = []
            for i, j in tempk.split(range(len(train_label))):
                kf.append([i, j])

            # tic = time.clock()
            if forest_index % 2 == 0:
                # print('rf')
                kfold = 0
                kfold_list = []
                for train_index, val_index in kf:
                    kfold += 1
                    clf = RandomForestClassifier(n_estimators=self.n_estimators,
                                                 n_jobs=-1, max_features="sqrt",
                                                 max_depth=self.max_depth,
                                                 min_samples_leaf=self.min_samples_leaf)
                    X_train = train_data[train_index, :]
                    y_train = train_label[train_index]
                    clf.fit(X_train, y_train)
                    kfold_list.append(clf)

                    X_val = train_data[val_index, :]
                    val_p = clf.predict_proba(X_val)
                    val_prob_forest[val_index, :] = val_p
            else:
                # print('ert')
                kfold = 0
                kfold_list = []
                for train_index, val_index in kf:
                    kfold += 1
                    clf = ExtraTreesClassifier(n_estimators=self.n_estimators,
                                               n_jobs=-1, max_features=1,
                                               max_depth=self.max_depth,
                                               min_samples_leaf=self.min_samples_leaf)
                    X_train = train_data[train_index, :]
                    y_train = train_label[train_index]
                    clf.fit(X_train, y_train)
                    kfold_list.append(clf)

                    X_val = train_data[val_index, :]
                    val_p = clf.predict_proba(X_val)
                    val_prob_forest[val_index, :] = val_p

            self.forest_list.append(kfold_list)

            val_prob[forest_index, :] = val_prob_forest

        val_avg = np.sum(val_prob, axis=0)
        val_avg /= self.num_forests
        val_concatenate = val_prob.transpose((1, 0, 2))
        val_concatenate = val_concatenate.reshape(val_concatenate.shape[0], -1)

        return [val_avg, val_concatenate]

    def train_and_predict(self, train_data, train_label, test_data):
        val_prob = np.zeros((self.num_forests, train_data.shape[0], self.num_classes), dtype=np.float64)
        predict_prob = np.zeros((self.num_forests, test_data.shape[0], self.num_classes), dtype=np.float64)
        n_dim = train_data.shape[1]

        for forest_index in range(self.num_forests):
            predict_prob_forest = np.zeros([test_data.shape[0], self.num_classes])
            val_prob_forest = np.zeros((train_data.shape[0], self.num_classes))

            tempk = KFold(n_splits=self.n_fold, shuffle=True)
            kf = []
            for i, j in tempk.split(range(len(train_label))):
                kf.append([i, j])

            # predict_p = np.zeros([test_data.shape[0], self.num_classes])
            # val_p = np.zeros([val_data.shape[0], self.num_classes])

            tic = time.clock()
            if forest_index % 2 == 1:
                # print("rf")
                kfold = 0
                for train_index, val_index in kf:
                    kfold += 1
                    clf = RandomForestClassifier(n_estimators=self.n_estimators,
                                             n_jobs=-1, max_features="sqrt",
                                             max_depth=self.max_depth,
                                             min_samples_leaf=self.min_samples_leaf)
                    X_train = train_data[train_index, :]
                    y_train = train_label[train_index]
                    clf.fit(X_train, y_train)

                    X_val = train_data[val_index, :]
                    val_p = clf.predict_proba(X_val)
                    val_prob_forest[val_index, :] = val_p

                    predict_p = clf.predict_proba(test_data)
                    predict_prob_forest += predict_p
                    toc = time.clock()
                    # print(toc - tic, ' s')
            else:
                # print("ert")
                kfold = 0
                for train_index, val_index in kf:
                    kfold += 1
                    clf = ExtraTreesClassifier(n_estimators=self.n_estimators,
                                               n_jobs=-1, max_features=1,
                                               max_depth=self.max_depth,
                                               min_samples_leaf=self.min_samples_leaf)

                    X_train = train_data[train_index, :]
                    y_train = train_label[train_index]
                    clf.fit(X_train, y_train)

                    X_val = train_data[val_index, :]
                    val_p = clf.predict_proba(X_val)
                    val_prob_forest[val_index, :] = val_p

                    predict_p = clf.predict_proba(test_data)
                    predict_prob_forest += predict_p
                    toc = time.clock()
                    # print(toc - tic, ' s')

            predict_prob_forest /= self.n_fold

            val_prob[forest_index, :] = val_prob_forest
            predict_prob[forest_index, :] = predict_prob_forest

        val_avg = np.sum(val_prob, axis=0)
        val_avg /= self.num_forests
        val_concatenate = val_prob.transpose((1, 0, 2))
        val_concatenate = val_concatenate.reshape(val_concatenate.shape[0], -1)

        predict_avg = np.sum(predict_prob, axis=0)
        predict_avg /= self.num_forests
        predict_concatenate = predict_prob.transpose((1, 0, 2))
        predict_concatenate = predict_concatenate.reshape(predict_concatenate.shape[0], -1)

        return [val_avg, val_concatenate, predict_avg, predict_concatenate]