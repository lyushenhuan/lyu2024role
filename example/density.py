import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree # type: ignore

def generateFeature(density, num_data, input_dim, grid_number=4):
    density = density.reshape(-1) / np.sum(density)
    # initialize
    feature = np.zeros((num_data, input_dim))
    # sample from density
    pt_id = np.random.choice(a=density.shape[0], size=(num_data, 1), p=density)
    for j in range(input_dim):
        feature[:,j:j+1] = pt_id - pt_id // grid_number * grid_number + np.random.uniform(low=0, high=1, size=(num_data, 1))
        pt_id //= grid_number
    return feature

def generateDensity(grid_num, input_dim, a):
    base_1 = np.array([1, a, a, 1])
    base_2 = np.array([1, a, a**2, 1])
    base_3 = np.array([1, a, a**3, 1])
    base_4 = np.array([1, a, a**4, 1])
    if input_dim == 2 and grid_num == 4:
        density_int = np.tensordot(base_1, base_2, 0)
    if input_dim == 3 and grid_num == 4:
        density_int = np.tensordot(np.tensordot(base_1, base_2, 0), base_3, 0)
    if input_dim == 4 and grid_num == 4:
        density_int = np.tensordot(np.tensordot(np.tensordot(base_1, base_2, 0), base_3, 0), base_4, 0)
    return density_int

def generateLabel(feature):
    # initialize
    label = np.zeros((feature.shape[0]))
    # get label
    for i in range(feature.shape[0]):
        label[i] = int(np.sum(feature[i] // 1)) % 2
    return label

if __name__=='__main__':
    # set hyperparameters
    grid_num = 4
    input_dim = 3 # if there are too many grids, file_name is too long and needs to be simplified
    num_data = 100000
    a = 3 # coefficient to construct density
    # generate density matrix
    density_int = generateDensity(grid_num, input_dim, a)
    # density = np.random.uniform(low=0, high=1, size=(grid_num, grid_num))
    density = density_int / np.sum(density_int)
    # generate data
    feature = generateFeature(density.reshape(-1), num_data, input_dim)
    # generate label
    label = generateLabel(feature)
    # train tree
    clf = tree.DecisionTreeClassifier(max_depth=6)
    clf.fit(feature, label)
    tree.plot_tree(clf)
    # save
    file_name = ''
    density_vec = density_int.reshape(-1)
    for i in range(density_vec.shape[0]):
        file_name += f'{str(int(density_vec[i]))}_'
    plt.savefig(f'./density/{file_name}.pdf')
