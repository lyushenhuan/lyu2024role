import argparse
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
    folder = f'./i={args.input_dimension}_gn={args.grid_number}_gw={args.grid_width}_f={args.feature_type}_d={args.data_number}'
    if not os.path.exists(folder):
        os.makedirs(folder)
    # save
    np.save(f'{folder}/feature', feature)
    np.save(f'{folder}/label', label)
    # plot
    plot(feature, label)

def generateFeature(args):
    # initialize
    feature = np.zeros((args.data_number, args.input_dimension))
    # generate random feature
    if args.feature_type == 'random':
        feature = np.random.uniform(low=0, high=args.range, size=feature.shape)
    # generate features on all grids
    elif args.feature_type == 'grid':
        for i in range(1, feature.shape[0]):
            feature[i] = feature[i-1]
            added = False
            j = feature.shape[1] - 1
            while not added:
                if feature[i][j] != args.range - 1:
                    feature[i][j] += 1
                    added = True
                else:
                    feature[i][j] = 0
                    j -= 1
        feature = feature + 0.5
    else:
        raise Exception(f'Feature generation is not implemented for feature_type = {args.feature_type}.')
    return feature

def generateLabel(feature, args):
    # initialize
    label = np.zeros((args.data_number, 1))
    for i in range(label.shape[0]):
        # constant grid width
        if args.grid_width == 'constant':
            label[i] = int(np.sum(feature[i] // 1)) % 2
        # increasing grid width
        elif args.grid_width == 'increasing':
            border = np.array([(i+1) * (i+2) // 2 for i in range(args.grid_number)])
            sum = 0
            for j in range(args.input_dimension):
                sum += np.min(np.where(border > feature[i][j])[0])
            label[i] = int(sum) % 2
        else:
            raise Exception(f'Label generation is not implemented for grid_width = {args.grid_width}.')
    return label

def plot(feature, label):
    # not plot if input dim is not 2
    if args.input_dimension != 2:
        print(f'Plot is not implemented for input_dimension = {args.input_dimension}.')
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
        plt.xlim(0, args.range)
        plt.ylim(0, args.range)
        plt.show()

def updataArgs(args):
    # calculate feature range
    if args.grid_width == 'constant':
        args.range = args.grid_number
    elif args.grid_width == 'increasing':
        args.range = args.grid_number * (args.grid_number + 1) // 2
    else:
        raise Exception(f'Feature range is not defined for grid_width = {args.grid_width}.')
    # calculate data_number when feature_type is grid
    if args.feature_type == 'grid':
        if args.data_number != -1:
            print(f'The argument data_number is ignored when feature_type is {args.feature_type}.')
        args.data_number = args.range ** args.input_dimension
    # raise exception when feature type is random and dataset number is not given
    if args.feature_type == 'random' and args.data_number == -1:
        raise Exception(f'data_number shold be given when feature_type = {args.feature_type}.')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Deep Tree')
    parser.add_argument('--input_dimension', default=10, type=int)
    parser.add_argument('--grid_number', default=4, type=int) # the number of grid in each dimension
    parser.add_argument('--grid_width', default='increasing', type=str, choices=['constant', 'increasing']) # constant (width=1), increasing (width=1, 2, 3,...), constant is hard to learn while increasing is easier
    parser.add_argument('--feature_type', default='random', type=str, choices=['random', 'grid'])
    parser.add_argument('--data_number', default=-1, type=int) # if feature type is grid, dataset number is the number of grids
    args = parser.parse_args()

    generateData(args)

# test
# python dataForDeepTree.py
# python dataForDeepTree.py --grid_number 4
# python dataForDeepTree.py --grid_number 4 --grid_width increasing
# python dataForDeepTree.py --grid_number 4 --grid_width increasing --feature_type random --data_number 1000
