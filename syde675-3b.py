from numpy import *
import numpy as np
import struct


def load_images(file_name):
    binfile = open(file_name, 'rb')
    buffers = binfile.read()
    magic, num, rows, cols = struct.unpack_from('>IIII',buffers, 0)
    bits = num * rows * cols
    images = struct.unpack_from('>' + str(bits) + 'B', buffers, struct.calcsize('>IIII'))
    binfile.close()
    images = np.reshape(images, [num, rows * cols])
    return images


global data
data = load_images('train-images.idx3-ubyte')


def pca(i):
    means = mean(data, axis=0)
    new_data = data-means
    covMat = np.cov(new_data.T)
    eigVals, eigVects = np.linalg.eig(covMat)
    n_eigValIndice = argsort(-eigVals)
    selectedfeature = np.matrix(eigVects.T[n_eigValIndice[:i]])
    finalData = new_data*selectedfeature.T
    finalData = finalData.real
    reconMat = (finalData*selectedfeature)+means
    return eigVals


eigvalue = sorted(pca(1), reverse=True)
eigvalue = np.real(eigvalue)
sum1 = []

for j in range(len(eigvalue)):
    sum1.append(eigvalue[j])
    if np.sum(sum1) > (np.sum(eigvalue)*0.95):
        print("Suitable d (POV=95%) is ", len(sum1))
        break
