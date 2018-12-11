from sklearn.model_selection import StratifiedKFold
import numpy as np
import Core
import scipy
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import Utility
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn import svm, metrics
from scipy.spatial import distance_matrix


def pseudo_error_silhouette(training_feature, training_label, classifier, testing_feature):
    classifier.fit(training_feature, training_label)
    testing_pseudo = classifier.predict(testing_feature)
    return -silhouette_score(testing_feature, testing_pseudo, random_state=1617)


def refine(A):
    '''
    Using gradient descent in JDA to refine A
    :param A: starting parameter
    :return: refined A
    '''

    # estimate the new label for target isntances
    Z = np.dot(A.T, Core.K)
    Z /= np.linalg.norm(Z, axis=0)
    Xs_new, Xt_new = Z[:, :Core.ns].T, Z[:, Core.ns:].T
    Ys = Core.Ys
    Core.classifier.fit(Xs_new, Ys)
    Y_tar_pseudo = Core.classifier.predict(Xt_new)

    X = Z
    ns, nt = Core.ns, Core.nt
    m, n = X.shape
    C = len(np.unique(Ys))
    H = np.eye(n) - 1.0 / n * np.ones((n, n))

    M = np.copy(Core.M0)
    N = 0
    for c in range(1, C + 1):
        e = np.zeros((n, 1))
        tt = Ys == c
        e[np.where(tt == True)] = 1.0 / len(Ys[np.where(Ys == c)])
        yy = Y_tar_pseudo == c
        ind = np.where(yy == True)
        inds = [item + ns for item in ind]
        if len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)]) ==0:
            e[np.isinf(e)] = 0
        else:
            e[tuple(inds)] = -1.0 / len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)])

        N = N + np.dot(e, e.T)

    M += N
    M = M / np.linalg.norm(M, 'fro')
    K = Core.K
    n_eye = Core.A_row
    a, b = np.linalg.multi_dot([K, M, K.T]) + Core.lamda * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
    w, V = scipy.linalg.eig(a, b)
    ind = np.argsort(w)
    A = V[:, ind[:Core.A_col]]

    return A


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
    Core.classifier.fit(Xs_new, Core.Ys)
    Yt_pseudo = Core.classifier.predict(Xt_new)

    Y = np.hstack((Core.Ys, Yt_pseudo))
    s_score = -silhouette_score(X, Y, random_state=1617)

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
    mmd = mmd_matrix.trace() / mmd_matrix.shape[0]
    # variance = (np.eye(Core.A_col)-np.linalg.multi_dot([A.T, Core.var_o, A])).trace()
    # cond_matrix = np.dot(A.T, A)-np.eye(Core.A_col)
    # cond = np.dot(cond_matrix, cond_matrix.T).trace()
    # print(mmd, variance, cond)

    print(mmd, s_score)
    fitness = mmd + s_score
    return fitness


def nfold_classification_error(features, labels, classifier, n_fold):
    '''
    Return the number of instances mis-classified by a classifier
    :param features:
    :param labels:
    :param classifier:
    :param n_fold:
    :return:
    '''
    error = 0
    skf = StratifiedKFold(n_splits=n_fold, random_state=1617)
    for trainIndex, testIndex in skf.split(features, labels):
        train_feature, test_feature = features[trainIndex], features[testIndex]
        train_label, test_label = labels[trainIndex], labels[testIndex]
        classifier.fit(train_feature, train_label)
        fold_error = (1.0-classifier.score(test_feature, test_label)) #*len(test_feature)
        error += fold_error
    error = error/n_fold
    return error


def fitness_function_error(A):
    # estimate the new label for target isntances
    Z = np.dot(A.T, Core.K)
    Z /= np.linalg.norm(Z, axis=0)
    Xs_new, Xt_new = Z[:, :Core.ns].T, Z[:, Core.ns:].T
    Ys = Core.Ys
    Core.classifier.fit(Xs_new, Ys)
    Y_tar_pseudo = Core.classifier.predict(Xt_new)

    X = Z
    ns, nt = Core.ns, Core.nt
    m, n = X.shape
    C = len(np.unique(Ys))
    H = np.eye(n) - 1.0 / n * np.ones((n, n))

    M = np.copy(Core.M0)
    N = 0
    for c in range(1, C + 1):
        e = np.zeros((n, 1))
        tt = Ys == c
        e[np.where(tt == True)] = 1.0 / len(Ys[np.where(Ys == c)])
        yy = Y_tar_pseudo == c
        ind = np.where(yy == True)
        inds = [item + ns for item in ind]
        e[tuple(inds)] = -1.0 / len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)])
        e[np.isinf(e)] = 0
        N = N + np.dot(e, e.T)

    M += N
    M = M / np.linalg.norm(M, 'fro')
    K = Core.K
    n_eye = Core.A_row
    a, b = np.linalg.multi_dot([K, M, K.T]) + Core.lamda * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
    w, V = scipy.linalg.eig(a, b)
    ind = np.argsort(w)
    A = V[:, ind[:Core.A_col]]

    Z = np.dot(A.T, Core.K)
    Z /= np.linalg.norm(Z, axis=0)
    Xs_new, Xt_new = Z[:, :Core.ns].T, Z[:, Core.ns:].T
    err = nfold_classification_error(Xs_new, Core.Ys, Core.classifier, n_fold=3)

    return err, A


def fitness_function_model(A):
    # estimate the new label for target isntances
    Z = np.dot(A.T, Core.K)
    Z /= np.linalg.norm(Z, axis=0)
    Xs_new, Xt_new = Z[:, :Core.ns].T, Z[:, Core.ns:].T
    Ys = Core.Ys
    Core.classifier.fit(Xs_new, Ys)
    Y_tar_pseudo = Core.classifier.predict(Xt_new)

    X = Z
    ns, nt = Core.ns, Core.nt
    m, n = X.shape
    C = len(np.unique(Ys))
    H = np.eye(n) - 1.0 / n * np.ones((n, n))

    M = np.copy(Core.M0)
    N = 0
    for c in range(1, C + 1):
        e = np.zeros((n, 1))
        tt = Ys == c
        e[np.where(tt == True)] = 1.0 / len(Ys[np.where(Ys == c)])
        yy = Y_tar_pseudo == c
        ind = np.where(yy == True)
        inds = [item + ns for item in ind]
        if len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)]) == 0:
            e[np.isinf(e)] = 0
        else:
            e[tuple(inds)] = -1.0 / len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)])
        N = N + np.dot(e, e.T)

    M += N
    M = M / np.linalg.norm(M, 'fro')
    K = Core.K
    n_eye = Core.A_row
    a, b = np.linalg.multi_dot([K, M, K.T]) + Core.lamda * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
    w, V = scipy.linalg.eig(a, b)

    ind = np.argsort(w)
    A = V[:, ind[:Core.A_col]]
    phi = [w[i] for i in ind[:Core.A_col]]
    phi = np.diag(phi)
    mmd_matrix = np.linalg.multi_dot([A.T, a, A])
    var_matrix = np.eye(A.shape[1]) - np.linalg.multi_dot([A.T, b, A])
    var_matrix = np.dot(var_matrix, phi)

    return mmd_matrix.trace()+var_matrix.trace(), A


def fitness_function_reverse(A):
    '''
    From A, calculate phi according to the lagrang, then use it to calcualte
    the fitness value
    :param A:
    :return:
    '''
    # estimate the new label for target isntances
    Z = np.dot(A.T, Core.K)
    Z /= np.linalg.norm(Z, axis=0)
    Xs_new, Xt_new = Z[:, :Core.ns].T, Z[:, Core.ns:].T
    Ys = Core.Ys
    Core.classifier.fit(Xs_new, Ys)
    Y_tar_pseudo = Core.classifier.predict(Xt_new)

    X = Z
    ns, nt = Core.ns, Core.nt
    m, n = X.shape
    C = len(np.unique(Ys))
    H = np.eye(n) - 1.0 / n * np.ones((n, n))

    M = np.copy(Core.M0)
    N = 0
    for c in range(1, C + 1):
        e = np.zeros((n, 1))
        tt = Ys == c
        e[np.where(tt == True)] = 1.0 / len(Ys[np.where(Ys == c)])
        yy = Y_tar_pseudo == c
        ind = np.where(yy == True)
        inds = [item + ns for item in ind]
        if len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)]) == 0:
            e[np.isinf(e)] = 0
        else:
            e[tuple(inds)] = -1.0 / len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)])
        N = N + np.dot(e, e.T)

    M += N
    M = M / np.linalg.norm(M, 'fro')
    K = Core.K
    n_eye = Core.A_row
    left = np.linalg.multi_dot([K, M, K.T]) + Core.lamda * np.eye(n_eye)
    left = np.dot(left, A)
    right = np.linalg.multi_dot([K, H, K.T])
    right = np.dot(right, A)
    # left = right * phi - b = a *X
    phi = np.linalg.lstsq(right, left, rcond=None)[0]
    mmd_matrix = np.dot(A.T, left)
    var_matrix = np.eye(A.shape[1]) - np.dot(A.T, right)
    var_matrix = np.dot(var_matrix, phi)

    return mmd_matrix.trace()+var_matrix.trace()


def classification_error(training_feature, training_label, classifier, testing_feature, testing_label):
    classifier.fit(training_feature, training_label)
    error = 1.0 - classifier.score(testing_feature, testing_label)
    return error


if __name__ == "__main__":
    v1 = np.asarray([-3, -2, 6])
    v2 = np.asarray([4, 5, -8])
