import numpy as np
import Utility
from sklearn.neighbors import KNeighborsClassifier


source = np.genfromtxt("data/Source", delimiter=",")
m = source.shape[1]-1
Xs = source[:, 0:m]
Ys = np.ravel(source[:, m:m + 1])
Ys = np.array([int(label) for label in Ys])

target = np.genfromtxt("data/Target", delimiter=",")
Xt = target[:, 0:m]
Yt = np.ravel(target[:, m:m + 1])
Yt = np.array([int(label) for label in Yt])

# normalize according to row
Xs /= np.linalg.norm(Xs, axis=1)[:, None]
Xt /= np.linalg.norm(Xt, axis=1)[:, None]
ns, nt = Xs.shape[0], Xt.shape[0]
n = ns + nt
C = len(np.unique(Ys))

# new dimension
k = 10
lamda = 1.0

# build kernel matrix
X = np.hstack((Xs.T, Xt.T))
kernel = 'primal'
K = Utility.kernel(kernel, X, None, gamma=0.5)
if kernel == 'primal':
    A_row, A_col = m, k
else:
    A_row, A_col = n, k


# build H matrix -> variant
H = np.eye(n) - 1.0 / n * np.ones((n, n))

# build M0 matrix
e = np.vstack((1.0 / ns * np.ones((ns, 1)), -1.0 / nt * np.ones((nt, 1))))
M0 = e * e.T * C

# original variant
var_o = np.linalg.multi_dot([K, H, K.T])

classifier = KNeighborsClassifier(n_neighbors=1)



if __name__ == "__main__":
    classifier = KNeighborsClassifier(n_neighbors=1)