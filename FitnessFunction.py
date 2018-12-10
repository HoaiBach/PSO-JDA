from sklearn.model_selection import StratifiedKFold
import numpy as np
import Core
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import Utility
from sklearn.metrics import silhouette_score
from sklearn import svm, metrics
from scipy.spatial import distance_matrix


def pseudo_error_silhouette(training_feature, training_label, classifier, testing_feature):
    classifier.fit(training_feature, training_label)
    testing_pseudo = classifier.predict(testing_feature)
    return -silhouette_score(testing_feature, testing_pseudo, random_state=1617)

def fitness_function(A):
    '''
    Calculate the fitness function of the matrix A
    :param A: A_row * A_col, defined in Core
    :return:
    '''
    # estimate the new label
    Z = np.dot(A.T, Core.K)
    Z /= np.linalg.norm(Z, axis=0)
    X = Z.T
    Xs_new, Xt_new = Z[:, :Core.ns].T, Z[:, :Core.nt].T
    return classification_error(Xs_new, Core.Ys, Core.classifier, Xt_new, Core.Yt)
    Core.classifier.fit(Xs_new, Core.Ys)
    Yt_pseudo = Core.classifier.predict(Xt_new)

    Y = np.hstack((Core.Ys, Yt_pseudo))
    s_score = -silhouette_score(X, Y , random_state=1617)

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
    B = np.linalg.multi_dot([Core.K, M, Core.K.T]) + Core.lamda * np.eye(Core.A_row)
    mmd_matrix = np.linalg.multi_dot([A.T, B, A])
    # now calculate the fitness function
    mmd = mmd_matrix.trace()/mmd_matrix.shape[0]
    # variance = (np.eye(Core.A_col)-np.linalg.multi_dot([A.T, Core.var_o, A])).trace()
    # cond_matrix = np.dot(A.T, A)-np.eye(Core.A_col)
    # cond = np.dot(cond_matrix, cond_matrix.T).trace()
    # print(mmd, variance, cond)

    print(mmd, s_score)
    fitness = mmd + s_score
    return fitness


def classification_error(training_feature, training_label, classifier, testing_feature, testing_label):
    classifier.fit(training_feature, training_label)
    error = 1.0 - classifier.score(testing_feature, testing_label)
    return error

if __name__ == "__main__":
    v1 = np.asarray([-3, -2, 6])
    v2 = np.asarray([4, 5, -8])
