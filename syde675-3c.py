from numpy import *
import numpy as np
import struct
import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error


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
    return reconMat, new_data


mse = []
length = 1
for j in range(1, 784, length):
    rec, data_aftermean = pca(j)
    MSE = mean_squared_error(data_aftermean, np.real(rec))
    mse.append(MSE)
plt.plot(mse)
plt.xlabel('d')
plt.ylabel('MSE')
plt.title('MSE against d')
plt.savefig('3c')
plt.show()
