import argparse
import density
import matplotlib.pyplot as plt
import numpy as np
import os

def generateData(args):
    # update args
    updataArgs(args)
    # generate feature
    feature = generateFeature(args)
    # generate label
    label = generateLabel(feature, args)
    # generate folder
    folder = f'./dataForDeepTree/i={args.input_dimension}_d={args.data_number}'
    if not os.path.exists(folder):
        os.makedirs(folder)
    # save
    np.save(f'{folder}/feature', feature)
    np.save(f'{folder}/label', label)

def generateFeature(args):
    # initialize
    feature = np.zeros((args.data_number, args.input_dimension))
    # generate random feature
    np.random.seed(0)
    # generate base tensor for density
    a = 3
    bases = np.array([[1/a**(i+1), 1/a**i, 1, 1/a**(i+1)] for i in range(args.input_dimension)])
    # generate density
    density_int = np.tensordot(bases[0], bases[1], 0)
    for i in range(2, args.input_dimension):
        density_int = np.tensordot(density_int, bases[i], 0)
    # generate feature
    feature = density.generateFeature(density_int.reshape(-1), args.data_number, args.input_dimension)
    return feature

def generateLabel(feature, args):
    # initialize
    label = np.zeros((args.data_number, 1))
    for i in range(label.shape[0]):
        # constant grid width
        label[i] = int(np.sum(feature[i] // 1)) % 2
    return label

def plot(feature, label):
    # not plot if input dim is not 2
    if feature.shape[1] != 2:
        print(f'Plot is not implemented for input_dimension = {feature.shape[1]}.')
    # plot
    else:
        for i in range(feature.shape[0]):
            if label[i] == 0:
                plt.plot(feature[i][0], feature[i][1], color='red', marker='o')
            elif label[i] == 1:
                plt.plot(feature[i][0], feature[i][1], color='blue', marker='o')
            else:
                raise Exception(f'Plot is not implemented for label = {label[i]}.')
        # set plot
        plt.xlim(0, np.max(feature))
        plt.ylim(0, np.max(feature))
        plt.show()

def updataArgs(args):
    # calculate feature range
    args.range = 4
    if args.data_number == -1:
        raise Exception(f'data_number shold be given.')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Dataset for Deep Tree')
    parser.add_argument('--input_dimension', default=10, type=int)
    parser.add_argument('--data_number', default=-1, type=int) # if feature type is grid, data number is the number of grids
    args = parser.parse_args()

    generateData(args)
