import argparse
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn import tree # type: ignore

class Dataset():
    def __init__(self, feature, label, args) -> None:
        # check the consistency of length
        if feature.shape[0] != label.shape[0]:
            raise Exception(f'The number of feature is {feature.shape[0]}, while the number of label is {label.shape[0]}.')
        # generate train/test feature/label
        self.num_data = feature.shape[0]
        self.train_feature = feature[0 : int(0.7 * self.num_data)]
        self.test_feature = feature[int(0.7 * self.num_data) : self.num_data]
        self.train_label = label[0 : int(0.7 * self.num_data)].reshape(-1)
        self.test_label = label[int(0.7 * self.num_data) : self.num_data].reshape(-1)

def loadData(args):
    feature = np.load(f'{args.folder}/feature.npy')
    label = np.load(f'{args.folder}/label.npy')
    dataset = Dataset(feature, label, args)
    return dataset

def main(args):
    max_size = 16
    # load data
    dataset = loadData(args)
    # train a single decesion tree
    for tree_size in range(1, max_size):
        trainTree(tree_size, dataset, args)
    # train a deep tree
    for depth in [2, 3, 4]:
        for tree_size in range(1, max_size):
            trainDeepTree(tree_size, depth, dataset, args)
    # train a random forest
    for width in [9, 19, 29]:
        for tree_size in range(1, max_size):
            trainRandomForest(tree_size, width, dataset, args)
    # spacing
    printAcc(0, 0, 0, args, spacing=True)

def printAcc(test_output, test_label, model_name, args, spacing=False):
    # generate accuracy string
    if spacing:
        acc_str = ''
    else:
        acc_str = f'The accuracy of {model_name} is {np.sum(test_output.reshape(-1) == test_label.reshape(-1)) / test_label.shape[0] * 100:.2f}%.'
        print(acc_str)
    # save to txt
    file_name = f'{args.folder}/acc.txt'
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            content = f.readlines()
    else:
        content = []
    with open(file_name, 'w') as f:
        content += f'{acc_str}\n'
        f.writelines(content)

def trainDeepTree(tree_size, depth, dataset, args):
    # initialize clfs
    clfs = []
    train_outputs = []
    # train
    for i in range(depth):
        # add augmented feature
        if i == 0:
            train_feature = dataset.train_feature
        else:
            train_feature = np.concatenate([dataset.train_feature, train_outputs[-1]], axis=1)
        # train a clf
        clf = tree.DecisionTreeClassifier(max_depth=tree_size)
        clf = clf.fit(train_feature, dataset.train_label)
        clfs.append(clf)
        train_outputs.append(clf.predict(train_feature).reshape(-1, 1))
    # test
    test_outputs = []
    for i in range(depth):
        # add augmented feature
        if i == 0:
            test_feature = dataset.test_feature
        else:
            test_feature = np.concatenate([dataset.test_feature, test_outputs[-1]], axis=-1)
        # forward a clf
        test_outputs.append(clfs[i].predict(test_feature).reshape(-1, 1))
    # calculate number of leaves
    n_leaves = 0
    for clf in clfs:
        n_leaves += clf.get_n_leaves()
    printAcc(test_outputs[-1], dataset.test_label, f'deepTree of size {tree_size} and depth {depth} ({n_leaves} leaves)', args)

def trainRandomForest(tree_size, width, dataset, args):
    # train
    clf = RandomForestClassifier(n_estimators=width, max_depth=tree_size)
    clf = clf.fit(dataset.train_feature, dataset.train_label)
    # test
    test_output = clf.predict(dataset.test_feature)
    # calcualte number of leaves
    n_leaves = 0
    for tree in clf.estimators_:
        n_leaves += tree.get_n_leaves()
    printAcc(test_output, dataset.test_label, f'randomForest of size {tree_size} and width {width} ({n_leaves} leaves)', args)

def trainTree(tree_size, dataset, args):
    # train
    clf = tree.DecisionTreeClassifier(max_depth=tree_size)
    clf = clf.fit(dataset.train_feature, dataset.train_label)
    # test
    test_output = clf.predict(dataset.test_feature)
    printAcc(test_output, dataset.test_label, f'singleTree of size {tree_size} ({clf.get_n_leaves()} leaves)', args)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Learn Deep Tree')
    parser.add_argument('--input_dimension', default=10, type=int)
    parser.add_argument('--data_number', default=-1, type=int) # if feature type is grid, data number is the number of grids
    args = parser.parse_args()

    args.folder = f'./dataForDeepTree/i={args.input_dimension}_d={args.data_number}'
    main(args)
