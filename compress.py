import matplotlib.pyplot as mpt
import numpy as np
from os import listdir , path ,mkdir
from os.path import isfile, join
import pca

def compress_images(DATA, k):
    output_dir="Output"
    isdir = path.isdir(output_dir)
    if(not isdir):
        mkdir(output_dir)

    Z = pca.compute_Z(DATA)
    COV = pca.compute_covariance_matrix(Z)
    L, PCS = pca.find_pcs(COV)
    Z_star = pca.project_data(Z, PCS, L, k, 0)
    print(Z_star.shape)
    height=60
    width=48

    image_reshape=np.array([[Z_star[i + j * width] for i in range(width)] for j in range(height)])
    print(image_reshape.shape)
    print(len(Z_star))
    # for i in range(len(Z_star)):
    #     out_file=output_dir+"/f"+str(i)+".png"
    #     print(out_file)
    #     mpt.imsave(out_file, Z[i][np.newaxis],cmap='gray')
    return 0

def load_data(input_dir):

    res= np.array([mpt.imread(input_dir+f).flatten() for f in listdir(input_dir)]).transpose()
    print(res.shape)
    return res
