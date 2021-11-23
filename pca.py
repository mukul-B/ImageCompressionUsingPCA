import numpy as np


def compute_Z(X, centering=True, scaling=False):
    X_mean = np.mean(X, axis=0)
    a = X - X_mean
    std = np.std(a, axis=0)
    scal = a / std
    # print(std,scal)
    # print(a)
    return scal


def compute_covariance_matrix(Z):
    return np.cov(Z.transpose(), ddof=False)


def find_pcs(COV):
    w, v = np.linalg.eig(np.array(COV))
    return w, v


def project_data(Z, PCS, L, k, var):
    b = PCS[:k].transpose()
    p = np.dot(Z, b)
    return p
