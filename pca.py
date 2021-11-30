import numpy as np


def compute_Z(X, centering=True, scaling=False):
    if centering:
        X_mean = np.mean(X, axis=0)
        a = X - X_mean
    else:
        a = X
    if scaling:
        std = np.std(a, axis=0)
        scal = a / std
    else:
        scal = a
    return scal


def compute_covariance_matrix(Z):
    return np.cov(Z.transpose(), ddof=False)


def find_pcs(COV):
    w, v = np.linalg.eig(np.array(COV))
    return w, v


def getK(L, var):
    totalV = L.sum()

    for i in range(len(L)):
        v = (L[:i].sum()) / totalV
        if v > var:
            break
    return i - 1


def project_data(Z, PCS, L, k, var):
    k = getK(L, var) if k == 0 else k

    b = PCS[:k].transpose()
    p = np.dot(Z, b)
    return p
