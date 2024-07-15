import numpy as np
import math
from sklearn.datasets import load_svmlight_file
import random
import pickle
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from .utils import *
from sklearn.model_selection import train_test_split

def data_shuffle(X_train, y_train):
    n_samples = X_train.shape[0]
    shuffle_ind = np.random.permutation(n_samples)
    X_train = X_train[shuffle_ind]
    y_train = y_train[shuffle_ind]
    return X_train, y_train

def set_fixed_random_seed(fixed_random_seed):
    np.random.seed(fixed_random_seed)
    random.seed(fixed_random_seed)

# def load_cifar100_dataset():
#     with open("../../dataset/cifar100emb/cifar100_train_emb.pkl", "rb") as f:
#         train_emb = pickle.load(f)
#         train_label = pickle.load(f)
#     # with open("../../dataset/cifar100emb/cifar100_test_emb.pkl", "rb") as f:
#     #     test_emb = pickle.load(f)
#     #     test_label = pickle.load(f)
#     train_emb, train_label, test_emb, test_label = train_val_split(train_emb, train_label,0.8)
#     nclist = [400]*50+[80]*50
#     train_emb, train_label=get_train_samples(train_emb,train_label,nclist)
#     print(train_emb.shape)
#     return train_emb, train_label, test_emb, test_label

def load_app_dataset(datasetname):
    with open("../../dataset/app_data/"+datasetname+".pkl", "rb") as f:
        data = pickle.load(f)
        label = pickle.load(f)
    if datasetname == 'acoustic':
        set_fixed_random_seed(42)
        X_train, y_train, test_data, test_label = train_test_split(data, label, test_size = 0.2, shuffle=True, stratify=label)
    elif datasetname == 'mirna':
        X_train, y_train, test_data, test_label = train_test_split(data, label, test_size = 0.15, shuffle=True, stratify=label)
    return X_train, y_train, test_data, test_label

def load_h_dataset(datasetname):
    datapath = '../../dataset/' + datasetname + "/"
    train_file = datapath + datasetname + '_train_data.pkl'
    test_file = datapath + datasetname + '_test_data.pkl'
    train_data, train_label, test_data, test_label = dataset_reader(train_file, test_file)
    return train_data, train_label, test_data, test_label

def load_vehicle_dataset(data_path):
    # print("dataset: vehicle")
    filename = data_path + 'vehicle.scale'
    (X, y) = load_svmlight_file(filename)
    X = X.toarray()
    y = np.int8(y - 1)

    X_train, X_test, y_train, y_test=train_test_split(X, y, stratify=y)

    return X_train, y_train, X_test, y_test

def load_usps_dataset(data_path):
    # print("dataset: ", datasetname)
    filename = data_path + 'usps'
    (X_train, y_train) = load_svmlight_file(filename)
    X_train = X_train.toarray()

    filename = data_path + 'usps.t'
    (X_test, y_test) = load_svmlight_file(filename)
    X_test = X_test.toarray()

    # X_train, X_test = train_test_normalization(X_train, X_test)
    y_train = np.int8(y_train - 1)
    y_test = np.int8(y_test - 1)

    return X_train, y_train, X_test, y_test

def load_phishing_dataset(data_path):
    # data_path = '../dataset/phishing/'
    print("dataset: phishing")
    filename = data_path + 'phishing'
    (X, y) = load_svmlight_file(filename)
    X = X.toarray()
    y = np.int8(y)
    X_train, X_test, y_train, y_test=train_test_split(X, y, stratify=y)
    # X_train, X_test = train_test_normalization(X_train, X_test)
    return X_train, y_train, X_test, y_test

def load_segment_dataset(data_path):
    # data_path = '../../dataset/segment/'
    # print("dataset: segment")
    filename = data_path + 'segment.scale'
    (X, y) = load_svmlight_file(filename)
    X = X.toarray()
    y = np.int8(y - 1)
    # print(len(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    # X_train, X_test = train_test_normalization(X_train, X_test)
    # print('all train dataset:', np.bincount(y_train))
    # X_train, y_train = get_train_samples(X_train, y_train, class_list)
    # X_test, y_test = get_test_samples(X_test, y_test, class_list)
    return X_train, y_train, X_test, y_test

def load_pendigits_dataset(data_path):
    datasetname="pendigits"
    # print("dataset: ", datasetname)
    filename = data_path + datasetname
    (X, y) = load_svmlight_file(filename)
    X = X.toarray()
    y = np.int8(y)
    # print('all training dataset',np.bincount(y))
    filename = data_path + datasetname + '.t'
    (X_test, y_test) = load_svmlight_file(filename)
    X_test = X_test.toarray()
    y_test = np.int8(y_test)
    # artificially imbalance
    # class_list = np.arange(500,49,-50)
    # class_list = np.arange(400, 39, -40)
    X_train, y_train = X, y
    # X_train, X_test = train_test_normalization(X_train, X_test)
    return X_train, y_train, X_test, y_test



# def load_usps_dataset(data_path):
#     datasetname = 'usps'
#     # print("dataset: ", datasetname)
#     filename = data_path + datasetname
#     (X_train, y_train) = load_svmlight_file(filename)
#     X_train = X_train.toarray()
#
#     filename = data_path + 'usps.t'
#     (X_test, y_test) = load_svmlight_file(filename)
#     X_test = X_test.toarray()
#
#     X_train, X_test = train_test_normalization(X_train, X_test)
#     y_train = np.int8(y_train - 1)
#     y_test = np.int8(y_test - 1)
#
#     # class_list = np.arange(800,49,-80)
#     class_list = [800,600,400,400,300,200,100,80,60,50]
#     # print(class_list)
#     # print(np.bincount(y_train))
#     X_train, y_train = get_train_samples(X_train, y_train, class_list)
#     X_test, y_test = get_test_samples(X_test, y_test, class_list)
#     return X_train, y_train, X_test, y_test

def load_letter_dataset():
    data_path = '../dataset/letter_libsvm/'
    datasetname = 'letter_libsvm'
    # print("dataset: ", datasetname)
    # filename = data_path + datasetname
    # (X_train, y_train) = load_svmlight_file(filename)
    # X_train = X_train.toarray()
    filename = data_path + 'letter.scale.t'
    (X_test, y_test) = load_svmlight_file(filename)
    X_test = X_test.toarray()

    # debug1 = uniquecounts(y_train)
    filename = data_path + 'letter.scale.val'
    (X_val, y_val) = load_svmlight_file(filename)
    X_val = X_val.toarray()
    filename = data_path + 'letter.scale.tr'
    (X_train1, y_train1) = load_svmlight_file(filename)
    X_train1 = X_train1.toarray()
    y_train= np.concatenate((y_val, y_train1), axis=0)
    X_train = np.vstack((X_val, X_train1))

    # X_train, y_train = X[:15000], y[:15000]
    # X_test, y_test = X[15000:], y[15000:]
    y_train = np.int8(y_train - 1)
    y_test = np.int8(y_test - 1)

    return X_train, y_train, X_test, y_test

def dataset_reader(train_file, test_file):

    f = open(train_file, 'rb')
    train_features = pickle.load(f)
    train_labels = pickle.load(f)
    f.close()

    f = open(test_file, 'rb')
    test_features = pickle.load(f)
    test_labels = pickle.load(f)
    f.close()

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    return train_features, train_labels, test_features, test_labels

def load_dataset(datasetname, data_path):
    h_list = ['car']
    my_list = ['vowel','letter','satimage','usps','vehicle','pendigits','segment','dna']
    if datasetname in h_list:
        X_train, y_train, X_test, y_test = load_h_dataset(datasetname)
    elif datasetname in ['acoustic','mirna']:
        X_train, y_train, X_test, y_test = load_app_dataset(datasetname)
    # elif datasetname == 'cifar100':
    #     X_train, y_train, X_test, y_test = load_cifar100_dataset()
    elif datasetname in my_list:
        # data_path = '../../dataset/'
        # data_path = data_path + datasetname + '/'
        if datasetname == 'dna60':
            X_train, y_train, X_test, y_test = load_dna60_dataset()
        elif datasetname == 'dna':
            X_train, y_train, X_test, y_test = load_dna_dataset()
        elif datasetname == 'letter':
            X_train, y_train, X_test, y_test = load_letter_dataset()
        elif datasetname == 'satimage':
            X_train, y_train, X_test, y_test = load_satimage_dataset(data_path)
        elif datasetname == 'usps':
            X_train, y_train, X_test, y_test = load_usps_dataset(data_path)
        elif datasetname == 'vehicle':
            X_train, y_train, X_test, y_test = load_vehicle_dataset(data_path)
        elif datasetname == 'segment':
            X_train, y_train, X_test, y_test = load_segment_dataset()
        elif datasetname == 'vowel':
            X_train, y_train, X_test, y_test = load_vowel_dataset(data_path)
        elif datasetname == 'pendigits':
            X_train, y_train, X_test, y_test = load_pendigits_dataset(data_path)
    return X_train, y_train, X_test, y_test

def train_test_normalization(X_train, X_test):
    X = np.vstack((X_train, X_test))
    min_d = np.min(X, 0)
    max_d = np.max(X, 0)
    range_d = max_d - min_d

    #segment.scale has one column 0.0
    tmp = range_d==0
    range_d[tmp]=1

    X_train -= min_d + 0.
    X_train /= range_d
    X_test -= min_d + 0.
    X_test /= range_d
    return X_train, X_test

def load_vowel_dataset(data_path):
    datasetname="vowel"
    # data_path = '../../dataset/'+datasetname+"/"
    # print("dataset: ", datasetname)
    # filename = data_path + datasetname
    filename = data_path + datasetname+'.scale'
    (X_train, y_train) = load_svmlight_file(filename)
    X_train = X_train.toarray()

    # filename = data_path + datasetname + '.t'
    filename = data_path + datasetname +'.scale' + '.t' #seems there's little truncation, almost no influence, but one point strange influence
    (X_test, y_test) = load_svmlight_file(filename)
    X_test = X_test.toarray()

    # X_train, X_test = train_test_normalization(X_train, X_test)

    y_train = np.int8(y_train)
    y_test = np.int8(y_test)
    print('all train dataset:',np.bincount(y_train))
    return X_train, y_train, X_test, y_test


def load_satimage_dataset(data_path):
    # data_path = 'drive/dataset/letter/letter-recognition.dataset'
    # data_path = '../../dataset/satimage/'
    # print("dataset: satimage")
    # filename = data_path+'satimage.scale.txt'
    # (X, y) = load_svmlight_file(filename)
    # X = X.toarray()
    #
    # X_train, y_train, X_test, y_test = train_test_split(X, y, 3104)
    # X_train, y_train = X[:3104], y[:3104]
    # X_test, y_test = X[3104:], y[3104:]

    filename = data_path + 'satimage.scale.tr'
    (X_train, y_train) = load_svmlight_file(filename)
    X_train = X_train.toarray()

    filename = data_path + 'satimage.scale.t'
    (X_test, y_test) = load_svmlight_file(filename)
    X_test = X_test.toarray()

    X = np.vstack((X_train, X_test))
    # y = np.concatenate((y_train, y_test), axis=0)

    y_train = np.int8(y_train - 1)
    y_test = np.int8(y_test - 1)
    # print('all train dataset:', np.bincount(y_train))
    return X_train, y_train, X_test, y_test

def load_dna_dataset():
    data_path = '../../dataset/dna/'
    # print("dataset: dna")
    filename = data_path + 'dna.scale.tr'
    (X_train, y_train) = load_svmlight_file(filename)
    X_train = X_train.toarray()

    filename = data_path + 'dna.scale.t'
    (X_test, y_test) = load_svmlight_file(filename)
    X_test = X_test.toarray()

    y_train = np.int8(y_train - 1)
    y_test = np.int8(y_test - 1)
    # print('total training samples:',np.bincount(y_train))
    return X_train, y_train, X_test, y_test

def load_dna60_dataset():
    data_path = '../../dataset/dna/'
    print("dataset: dna60")
    filename = data_path + 'dna.scale.tr'
    (X_train, y_train) = load_svmlight_file(filename)
    X_train = X_train.toarray()

    filename = data_path + 'dna.scale.t'
    (X_test, y_test) = load_svmlight_file(filename)
    X_test = X_test.toarray()

    X_train = X_train[:, 60:120]
    X_test = X_test[:, 60:120]

    y_train = np.int8(y_train - 1)
    y_test = np.int8(y_test - 1)
    return X_train, y_train, X_test, y_test

def load_adult_dataset():
    data_path = '../dataset/adult/'
    print("dataset: adult")

    X_train = np.loadtxt(data_path+'train.txt')
    y_train = np.loadtxt(data_path+'label_train.txt')
    X_test = np.loadtxt(data_path+'test.txt')
    y_test = np.loadtxt(data_path+'label_test.txt')

    X = np.vstack((X_train, X_test))
    # y = np.concatenate((y_train, y_test), axis=0)

    min_d = np.min(X, 0)
    max_d = np.max(X, 0)
    range_d = max_d - min_d
    X_train -= min_d + 0.
    X_train /= range_d
    X_test -= min_d + 0.
    X_test /= range_d

    y_train = np.int8(y_train)
    y_test = np.int8(y_test)
    return X_train, y_train, X_test, y_test

def load_a9a_dataset():
    data_path = '../dataset/a9a/'
    print("dataset: a9a")
    filename = data_path + 'a9a.txt'
    (X, y) = load_svmlight_file(filename)
    X = X.toarray()
    y = (y + 1) / 2

    X_train, y_train = X[:32561], y[:32561]
    X_test, y_test = X[32561:], y[32561:]

    y_train = np.int8(y_train)
    y_test = np.int8(y_test)
    return X_train, y_train, X_test, y_test



def generate_mease_dataset(n_samples, effective_dimension, dimension):
    q=0.05 #bayes error
    dim = dimension
    J_dim = effective_dimension
    X = np.random.random((n_samples, dim))
    X_J = X[:,:J_dim]
    Sum_X_J = np.sum(X_J, axis = 1)
    Ind=np.zeros(n_samples)
    Ind[Sum_X_J>(J_dim/2)]=1
    Y_prob=q+(1-2*q)*Ind
    Y_rand = np.random.random((n_samples,))
    Y=np.zeros((n_samples, ))
    Y[Y_rand<Y_prob]=1
    Y = np.int8(Y)
    return X, Y

def main():
    # x_train, y_train, X_test, y_test  = load_vehicle_dataset()
    # x_train, y_train, X_test, y_test = load_segment_dataset()
    # x_train, y_train, X_test, y_test = load_vowel_dataset()
    # x_train, y_train, X_test, y_test = load_pendigits_dataset()
    # x_train, y_train, x_test, y_test = load_usps_dataset()
    # x_train, y_train, X_test, y_test = load_adult_dataset()
    # x_train, y_train, X_test, y_test = load_a9a_dataset()
    # x_train, y_train, X_test, y_test = load_phishing_dataset()
    x_train, y_train, X_test, y_test = load_satimage_dataset()
    x_train = 1

if __name__ == "__main__":
    main()
