import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from src.gcForest import gcForest
from sklearn.datasets import load_svmlight_file
from src.utils import *
from src.load_dataset import *
import time
import argparse
import random
from sklearn.metrics import accuracy_score

def main():
    print("**********", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),"**********")

    # n_estimators = 1000  # number of trees in each forest
    # n_size = 64
    for n_estimators in [50, 100, 200, 400, 800, 1600]: #50, 100, 200, 400, 800,
        for n_size in [8, 16, 32]:
            n_forests = 2
            fixed_random_seed = 64
            # print("fixed_random_seed =", fixed_random_seed)
            np.random.seed(fixed_random_seed)
            random.seed(fixed_random_seed)

            # X_train, y_train, X_test, y_test = load_letter_dataset()
            X_train, y_train, X_test, y_test = load_phishing_dataset('../dataset/phishing/')
            print(len(X_train),len(y_test),X_train.shape[1],len(np.bincount(y_train)))

            print("tree_depth = {}, n_estimators = {}, num_forests = {}".format(n_size, n_estimators, n_forests))

            n_classes = int(np.max(y_train) + 1)

            gc = gcForest(n_estimators, n_forests, n_classes, 2, n_size, 5)
            val_p, val_acc, best_layer_index = gc.train(X_train, y_train)
            test_prob = gc.predict(X_test)
            print(val_acc)
            print('best layer:', best_layer_index)
            print(compute_accuracy(y_test,test_prob))

            # # or use the following train_and_predict, which discards each layer after evaluating and saves the memory:
            # gc = gcForest(n_estimators, n_forests, n_classes, 31, 100, 3)
            # val_p, val_acc, test_p, test_acc, best_layer_index = gc.train_and_predict(X_train, y_train, X_test, y_test)
            # print(test_acc,test_acc[best_layer_index])

if __name__ == '__main__':
    main()