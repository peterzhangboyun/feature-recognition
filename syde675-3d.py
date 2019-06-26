from numpy import *
import numpy as np
import struct
import os
import matplotlib.pylab as plt
from pylab import cm


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
    return reconMat


def load_mnist_train(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


X_train, Y_train = load_mnist_train('', kind='train')
fig, ax = plt.subplots(nrows=1, ncols=6)


img = X_train[Y_train == 5][1].reshape(28, 28)
img1 = data[0].reshape(28, 28)
ax[0].imshow(img1, cmap=cm.gray)
ax[0].set_title('origin')
img_mtx = img.reshape(784, 1)
rec_img = []
pca_new_d = [1, 10, 50, 250, 784]


for i in range(len(pca_new_d)):
    img_rec = pca(pca_new_d[i]).real
    img_pca = (img_rec[0]).reshape(28, 28)
    ax[i+1].imshow(img_pca, cmap=cm.gray)
    ax[i+1].set_title('d=%d' % pca_new_d[i])

plt.savefig('3d')
plt.show()
