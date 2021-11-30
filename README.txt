compress_images
1) the output directory is created if it is not present
2) compute_Z
    if centering flag is on , np mean is subracted from sample space
    if scaling flag is on , np standard deviation is dived from sample space
3) compute_covariance_matrix
    using np.cov the covariance is calculated
4) find_pcs
    np.linalg.eig gives us normalized eigen vectors ordered in order of decreasing eigen values
5) project_data
    the top k pca is then dot product with modified sample space
6) the project data is restored by multipling it with pca
7) finally images are written using matplotlib.pyplot.imsave