from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from numpy import *
import numpy as np
import struct
# np.seterr(invalid='ignore')


def load_images(file_name):
    binfile = open(file_name, 'rb')
    buffers = binfile.read()
    magic, num, rows, cols = struct.unpack_from('>IIII',buffers, 0)
    bits = num * rows * cols
    images = struct.unpack_from('>' + str(bits) + 'B', buffers, struct.calcsize('>IIII'))
    binfile.close()
    images = np.reshape(images, [num, rows * cols])
    # images = images.T
    return images


def load_labels(file_name):
    binfile = open(file_name, 'rb')
    buffers = binfile.read()
    magic, num = struct.unpack_from('>II', buffers, 0)
    labels = struct.unpack_from('>' + str(num) + "B", buffers, struct.calcsize('>II'))
    binfile.close()
    labels = np.reshape(labels, [num])
    return labels


test_data = load_images('t10k-images.idx3-ubyte')
test_data = np.float32(test_data)
test_label = load_labels('t10k-labels.idx1-ubyte')
test_label = np.float32(test_label)
training_data = load_images('train-images.idx3-ubyte')
training_data = np.float32(training_data)
training_label = load_labels('train-labels.idx1-ubyte')
training_label = np.float32(training_label)


def PCA_algorithm_a(training_data, training_label, test_data, test_label, d):
    pca = PCA(n_components=d)
    pca.fit(training_data)
    pca.fit(test_data)
    final_training = pca.transform(training_data)
    final_test = pca.transform(test_data)
    return final_training, final_test


def KNN(training_data, training_label, test_data, test_label, n):
    neigh = KNeighborsClassifier(n_neighbors=n)
    neigh.fit(training_data, training_label)
    test_result = neigh.predict(test_data)
    test_score = neigh.score(test_data, test_label)
    return test_result, test_score


a, b = PCA_algorithm_a(training_data, training_label, test_data, test_label, 5)
print(KNN(a, training_label, b, test_label, 1)[1])

print(KNN(training_data, training_label, test_data, test_label, 1)[1])


# import numpy as np
# from scipy import stats
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import scipy.linalg as la
# import pprint
# from scipy.spatial import distance
# from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
#
# np.seterr(divide='ignore', invalid='ignore')
#
# # load train_data
# with open('train-images.idx3-ubyte', 'rb') as f:
#     train_images = extract_images(f)
# train_data= np.ones((60000, 784))
# for i in range(len(train_images)):
#     slice = ( train_images[i].reshape(784, 1).T )
#     train_data[i] = slice
#
# # load train_label
# with open('train-labels.idx1-ubyte', 'rb') as l:
#     train_labels = extract_labels(l)
# train_label= np.ones( (60000,1) )
# for i in range(len(train_labels)):
#     slice = ( train_labels[i].reshape(1, 1).T )
#     train_label[i] = slice
#
# # load test_data
# with open('t10k-images.idx3-ubyte', 'rb') as f:
#     test_images = extract_images(f)
# test_data= np.ones( (10000,784) )
# for i in range(len(test_images)):
#     slice = ( test_images[i].reshape(784, 1).T )
#     test_data[i] = slice
#
# # load test_label
# with open('t10k-labels.idx1-ubyte', 'rb') as l:
#     test_labels = extract_labels(l)
# test_label= np.ones( (10000,1) )
# for i in range(len(test_labels)):
#     slice = ( test_labels[i].reshape(1, 1).T )
#     test_label[i] = slice
#
# # calculate the distance matrix of train_data & test_data, return the index of k_biggest_value
# def dstc(data_a, data_b, k):
#     for i in range(len(data_a.T)):                          # normalize test data
#         diff = max(data_a.T[i]) - min(data_a.T[i])
#         data_a.T[i] /= diff
#
#     for i in range(len(data_b.T)):                          # normalize train data
#         diff = max(data_b.T[i]) - min(data_b.T[i])
#         data_b.T[i] /= diff
#
#     dstc_mtx = distance.cdist(data_a, data_b, 'euclidean')  # distance
#
#     dstc_mtx_bg = np.argsort(dstc_mtx, axis=1)              # sort distance
#     bg_num = dstc_mtx_bg[:,0:k]                             # select k_min distance
#     return bg_num
#
# # predict class label with given index, compare with true label
# def classify(test_label, train_label, biggest_k):
#     cls,n = [], 0
#     for i in biggest_k:                                     # read index in train_label
#         label_group=[]
#         for j in range(len(i)):
#             label_group.append( int(train_label[i[j]]) )
#         mode = (stats.mode(label_group))                    # find mode of labels
#         cls.append (mode[0])
#
#     for i in range(600):
#         if test_label[i] == cls[i]:                         # judge labels
#             n+=1
#     return round(n/600,4)
#
# # reconstruct MNIST using PCA
# def pca(data,d):
#     pca = PCA( n_components=d )
#     pca.fit_transform(data)
#     eigenVectors = ( pca.components_ )
#     pro_data = np.dot(data, eigenVectors.T)
#     rec_data = np.dot(pro_data, eigenVectors)
#     return rec_data
#
# print('_______________')
# # question {a}
# k = [1,3,5,11]
#
# for i in [1,3,5,11]:
#     biggest_k = dstc(test_data[0:600], train_data[0:600], i)    # get the index of k_biggest sample
#     print('k = '+str(i)+',',classify(test_label[0:600], train_label[0:600], biggest_k))
#
# print('_______________')
# # question {b}
# d, result = [5, 50, 100, 500], []
# train_data = np.nan_to_num(train_data)                      # replace Nan
#
# for i in d:
#     rec_data = pca(train_data,i)                            # reconstruct the data
#     for j in k:
#         biggest_k = ( dstc(test_data[0:600], rec_data[0:600], j) )
#         result.append( classify( test_label[0:600], train_label[0:600], biggest_k ) )
# print( np.array(result).reshape(4,4) )
