import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams['font.size'] = 16

def plot(model_list, acc_all, size_all, folder):
    # plot style
    # color_list = ['#1DA462', '#DD5144', '#FFCD46', '#4C8BF5', '#DD5144', '#FFCD46', '#4C8BF5']
    color_list = ['green', 'red', 'blue', 'darkviolet', 'red', 'blue', 'darkviolet']
    linestyle_list = ['dotted', 'solid', 'solid', 'solid', (0, (5, 3)), (0, (5, 3)), (0, (5, 3))]
    # plot all model
    for i in range(len(model_list)):
        acc_i = np.array([acc_all[j][i] for j in range(len(acc_all))]) / 100
        acc_mean = np.mean(acc_i, axis=0)
        acc_std = np.std(acc_i, axis=0) if len(acc_i) > 1 else np.zeros_like(acc_mean)
        plotModel(model_list[i], acc_mean, acc_std, size_all[0][i], color_list[i], linestyle_list[i])
    # set plot
    plt.xlabel('Number of Leaves')
    plt.ylabel('Test Accuracy')
    plt.xscale('log')
    plt.legend()
    dim = folder.split('i=')[1][0]
    plt.savefig(f'{folder}/acc_{dim}d.pdf', bbox_inches = 'tight')
    plt.savefig(f'{folder}/acc_{dim}d.png', bbox_inches = 'tight')
    plt.show()

def plotModel(model, y, std, x, color, linestyle):
    plt.plot(x, y, label=model[2], color=color, linestyle=linestyle, linewidth=2)
    plt.fill_between(x, y - std, y + std, color=color, alpha=0.1)

def readAcc(file_path, model_list):
    # initialize
    acc_list = []
    size_list = [] # record the number of leaves
    # read accuracy
    with open(file_path, 'r') as f:
        content = f.readlines()
    # find accuracy of each model
    for model in model_list:
        acc = []
        size = []
        for line in content:
            if model[0] in line and model[1] in line:
                acc.append(float(line.split('is ')[1].split('%')[0]))
                size.append(int(line.split('(')[1].split(' ')[0])) # number of leaves
        acc_list.append(acc)
        size_list.append(size)
    return acc_list, size_list

if __name__=='__main__':
    # set plot lines
    model_list = [
        ['singleTree', '', 'T'], 
        ['deepTree', 'depth 2', 'DT-2'],
        ['deepTree', 'depth 3', 'DT-3'],
        ['deepTree', 'depth 4', 'DT-4'],
        ['randomForest', 'width 9', 'RF-9'],
        ['randomForest', 'width 19', 'RF-19'],
        ['randomForest', 'width 29', 'RF-29']
    ]
    # file path
    folder_list = ['i=2_d=1000000',
                   'i=4_d=1000000',
                   'i=8_d=1000000']
    for folder in folder_list:
        folder = f'./dataForDeepTree/{folder}'
        file_path = f'{folder}/acc.txt'
        # initialize acc and size list
        acc_all = []
        size_all = []
        # read accuracy of each model with different tree size
        acc_list, size_list = readAcc(file_path, model_list)
        acc_all.append(acc_list)
        size_all.append(size_list)
        # plot accuracy
        plot(model_list, acc_all, size_all, folder)
