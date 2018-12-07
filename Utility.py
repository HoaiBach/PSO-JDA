'''
Created on 24/09/2018

@author: nguyenhoai2
'''
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import euclidean_distances as ecd

def normalize_src_tar(src_fea, tar_fea):
    src_fea_new = np.copy(src_fea)
    tar_fea_new = np.copy(tar_fea)
    all_fea_new = np.append(src_fea_new, tar_fea_new, axis=0)
    max_fea = np.max(all_fea_new, axis=0)
    min_fea = np.min(all_fea_new, axis=0)

    for col in range(len(src_fea_new[0])):
        max_value = max_fea[col]
        min_value = min_fea[col]

        for row in range(len(src_fea_new)):
            if(max_value == min_value):
                src_fea_new[row,col] = 1
            else:
                src_fea_new[row,col] = (src_fea_new[row,col]-min_value)/(max_value-min_value)

        for row in range(len(tar_fea_new)):
            if(max_value == min_value):
                tar_fea_new[row,col] = 1
            else:
                tar_fea_new[row,col] = (tar_fea_new[row,col]-min_value)/(max_value-min_value)
    
    return src_fea_new, tar_fea_new


def standarize_src_tar(src_fea, tar_fea):
    src_fea_new = np.copy(src_fea)
    tar_fea_new = np.copy(tar_fea)
    all_fea_new = np.append(src_fea_new, tar_fea_new, axis=0)

    stand_data = standarize_data(all_fea_new)
    stand_src = stand_data[0:len(src_fea_new), :]
    stand_tar = stand_data[len(src_fea_new): len(src_fea_new)+len(tar_fea_new), :]

    return stand_src, stand_tar


# standarize data
def standarize_data(data):
    data_new = np.copy(data)
    std = np.std(data_new, axis=0)
    mean = np.mean(data_new, axis=0)

    for row in range(len(data_new)):
        for col in range(len(data_new[0])):
            std_value = std[col]
            mean_value = mean[col]

            if std_value==0:
                data_new[row, col] = 1
            else:
                data_new[row, col] = (data_new[row, col] - mean_value)/std_value

    return data_new


def normalize_square_row(data):
    data_new = np.copy(data)
    denominators = np.sqrt(np.sum(np.square(data), axis=0))
    data_new = data_new/denominators
    return data_new


def normalize_square_column(data):
    data_new = np.copy(data)
    denominators = np.sqrt(np.sum(np.square(data), axis=1))
    data_new = data_new/denominators[:, None]
    return data_new


# Select sample_size instances from each class
# In total, there will be sample_size*y samples selected
def balanced_sample_maker(X, y, sample_size, random_seed=1617):
    uniq_levels = np.unique(y)
    uniq_counts = {level: sum(y == level) for level in uniq_levels}

    if not random_seed is None:
        np.random.seed(random_seed)

    # find observation index of each class levels
    groupby_levels = {}
    for ii, level in enumerate(uniq_levels):
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_levels[level] = obs_idx
    # oversampling on observations of each label
    balanced_copy_idx = []
    for gb_level, gb_idx in groupby_levels.items():
        over_sample_idx = np.random.choice(gb_idx, size=sample_size, replace=True).tolist()
        balanced_copy_idx+=over_sample_idx
    np.random.shuffle(balanced_copy_idx)

    data_train=X[balanced_copy_idx]
    labels_train=y[balanced_copy_idx]
    if  ((len(data_train)) == (sample_size*len(uniq_levels))):
        print('number of sampled example ', sample_size*len(uniq_levels), 'number of sample per class ', sample_size, ' #classes: ', len(list(set(uniq_levels))))
    else:
        print('number of samples is wrong ')

    labels, values = zip(*Counter(labels_train).items())
    print('number of classes ', len(list(set(labels_train))))
    check = all(x == values[0] for x in values)
    print(check)
    if check == True:
        print('Good all classes have the same number of examples')
    else:
        print('Repeat again your sampling your classes are not balanced')

    return data_train, labels_train


def kernel(ker, X, X2, gamma):
    if not ker or ker == 'primal':
        return X
    elif ker == 'linear':
        if not X2:
            K = np.dot(X.T, X)
        else:
            K = np.dot(X.T, X2)
    elif ker == 'rbf':
        n1sq = np.sum(X ** 2, axis=0)
        n1 = X.shape[1]
        if not X2:
            D = (np.ones((n1, 1)) * n1sq).T + np.ones((n1, 1)) * n1sq - 2 * np.dot(X.T, X)
        else:
            n2sq = np.sum(X2 ** 2, axis=0)
            n2 = X2.shape[1]
            D = (np.ones((n2, 1)) * n1sq).T + np.ones((n1, 1)) * n2sq - 2 * np.dot(X.T, X)
        K = np.exp(-gamma * D)
    elif ker == 'sam':
        if not X2:
            D = np.dot(X.T, X)
        else:
            D = np.dot(X.T, X2)
        K = np.exp(-gamma * np.arccos(D) ** 2)
    return K


# get median distance from a dataset
def medianDistance(dataset):
    pwDistance = ecd(dataset)
    return np.median(np.asarray(pwDistance))

if __name__ == "__main__":
    data = np.asarray([[3,4],[1,2]])
    print(normalize_square_row(data))