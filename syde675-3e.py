from numpy import *
import numpy as np
import struct
import matplotlib.pylab as plt


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


eigvalue_after = sorted(pca(1), reverse=True)
eigvalue_after = np.real(eigvalue_after)
plt.plot(eigvalue_after)
plt.xlabel('Number of Eigenvectors(d)')
plt.ylabel('Eigenvalue')
plt.xlim(0, 784)
plt.ylim(-1.33569988e-11, 3.4528143e+05)
plt.xticks(np.arange(0, 784, 100))
ax = plt.gca()
ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
plt.savefig('3e')
plt.show()
