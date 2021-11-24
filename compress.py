from os import listdir, path, mkdir
import matplotlib.pyplot as mpt
import numpy as np
import pca


def compress_images(DATA, k):
    output_dir = "Output"
    isdir = path.isdir(output_dir)
    if not isdir:
        mkdir(output_dir)
    Z = pca.compute_Z(DATA)
    COV = pca.compute_covariance_matrix(Z)
    L, PCS = pca.find_pcs(COV)
    # print(pca.getK(L, 0.951))
    Z_star = pca.project_data(Z, PCS, L, k, 0)
    z_compression = np.matmul(Z_star, PCS.transpose()[:k]).transpose()
    for i in range(len(z_compression)):
        out_file = output_dir + "/f" + str(i) + ".png"
        image_reshape = realign(z_compression[i])
        mpt.imsave(out_file, image_reshape, cmap='gray')
    return 0


def realign(flat_img):
    height = 60
    width = 48
    min = flat_img.min()
    max = flat_img.max()
    image_reshape = np.array(
        [[255 * (flat_img[h * width + w] - min) / (max - min) for w in range(width)] for h in range(height)])
    return image_reshape


def load_data(input_dir):
    res = np.array([mpt.imread(input_dir + f).flatten().astype(float) for f in listdir(input_dir)]).transpose()
    return res
