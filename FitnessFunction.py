from sklearn.model_selection import StratifiedKFold
import numpy as np
import Core
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import Utility
from sklearn.metrics import silhouette_score
from sklearn import svm, metrics
from scipy.spatial import distance_matrix


def fitness_function(A):
    '''
    Calculate the fitness function of the matrix A
    :param A: n * k, where m is the original dimension, k is the new dimesion
    :return:
    '''
    # estimate the new label
    Z = np.dot(A.T, Core.K)
    Z /= np.linalg.norm(Z, axis=0)
    Xs_new, Xt_new = Z[:, :Core.ns].T, Z[:, :Core.nt].T
    Core.classifier.fit(Xs_new, Core.Ys)
    Yt_pseudo = Core.classifier.predict(Xt_new)

    # now build M
    N = 0
    for c in range(1, Core.C + 1):
        e = np.zeros((Core.n, 1))
        tt = Core.Ys == c
        e[np.where(tt == True)] = 1.0 / len(Core.Ys[np.where(Core.Ys == c)])
        yy = Yt_pseudo == c
        ind = np.where(yy == True)
        inds = [item + Core.ns for item in ind]
        if len(Yt_pseudo[np.where(Yt_pseudo == c)]) != 0:
            e[tuple(inds)] = -1.0 / len(Yt_pseudo[np.where(Yt_pseudo == c)])
        else:
            e[np.isinf(e)] = 0
        N = N + np.dot(e, e.T)
    M = Core.M0 + N
    M = M / np.linalg.norm(M, 'fro')
    B = np.linalg.multi_dot([Core.K, M, Core.K.T]) + Core.lamda * np.eye(Core.n)

    # now calculate the fitness function
    fitness = np.linalg.multi_dot([A.T, B, A]).trace() / np.linalg.multi_dot([A.T, Core.var_o, A]).trace()
    return fitness


if __name__ == "__main__":
    v1 = np.asarray([-3, -2, 6])
    v2 = np.asarray([4, 5, -8])
