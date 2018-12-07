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

# build kernel matrix
X = np.hstack((Xs.T, Xt.T))
K = Utility.kernel('rbf', X, None, gamma=2.0)

# build H matrix -> variant
H = np.eye(n) - 1 / n * np.ones((n, n))

# build M0 matrix
e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
M0 = e * e.T * C

# original variant
var_o = np.linalg.multi_dot([K, H, K.T])

classifier = KNeighborsClassifier(n_neighbors=1)

# new dimension
k = 30
lamda = 1.0

if __name__ == "__main__":
    classifier = KNeighborsClassifier(n_neighbors=1)