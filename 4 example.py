from numpy import *
from sklearn.decomposition import PCA
import numpy as np
import numpy as np
import struct
import os
import matplotlib.pyplot as plt
from pylab import imshow,show,cm

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


global dataMat
# data = load_images('train-images.idx3-ubyte')
dataMat=load_images('t10k-images.idx3-ubyte')
print(shape(dataMat))

# def pca(i):
#     means = mean(data, axis=0)
#     new_data = data-means
#     covMat = np.cov(new_data.T)
#     eigVals, eigVects = np.linalg.eig(covMat)
#     n_eigValIndice = argsort(-eigVals)
#     selectedfeature = np.matrix(eigVects.T[n_eigValIndice[:i]])
#     finalData = new_data*selectedfeature.T
#     finalData = finalData.real
#     reconMat = (finalData*selectedfeature)+means
#     return finalData

def pca(topNfeat):
    global redEigVects
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals  # remove mean
    covMat = cov(meanRemoved, rowvar=0)
    eigVals, eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)  # sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat + 1):-1]  # cut off unwanted dimensions
    redEigVects = eigVects[:, eigValInd]  # reorganize eig vects largest to smallest
    # 得到低维度数据
    lowDDataMat = meanRemoved * redEigVects  # transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat


def KNN(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)  # axis=0, 表示列。axis=1, 表示行。
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# import numpy as np
# from sklearn import datasets
#
# iris = datasets.load_iris()
# iris_X = iris.data
# iris_y = iris.target
# np.unique(iris_y)
# # Split iris data in train and test data
# # A random permutation, to split the data randomly
# np.random.seed(0)
# # permutation随机生成一个范围内的序列
# indices = np.random.permutation(len(iris_X))
# # 通过随机序列将数据随机进行测试集和训练集的划分
# iris_X_train = iris_X[indices[:-10]]
# iris_y_train = iris_y[indices[:-10]]
# iris_X_test = iris_X[indices[-10:]]
# iris_y_test = iris_y[indices[-10:]]
# # Create and fit a nearest-neighbor classifier
# from sklearn.neighbors import KNeighborsClassifier
#
# knn = KNeighborsClassifier()
# knn.fit(iris_X_train, iris_y_train)
#
# KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#                      metric_params=None, n_jobs=1, n_neighbors=5, p=2,
#                      weights='uniform')
# knn.predict(iris_X_test)
# print(iris_y_test)


def kNN(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[1]
    diff = np.tile(newInput, (numSamples, 1)).T - dataSet
    squaredDiff = diff ** 2
    squaredDist = np.sum(squaredDiff, axis=1)
    distance = squaredDist ** 0.5
    sortedDistIndices = np.argsort(distance)
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
    return maxIndex

# %% [markdown]
# ## Q4.a
# k = [1, 3, 5, 11]
# testingImages = load_images('t10k-images.idx3-ubyte')
# testLabels = load_labels('t10k-labels.idx1-ubyte')
# trainingImages = load_images('train-images.idx3-ubyte')
# trainingLabels = load_labels('train-labels.idx1-ubyte')
# print(shape(testingImages))
# print(shape(testLabels))
# print(shape(trainingImages))
# print(shape(trainingLabels))

# %% [markdown]
# ## Q4.a
# # k = [1, 3, 5, 11]
# k=[1]

# accuracys = []
# for j in k:
#     accuracy = 0
#     # for i in range(testingImages.shape[1]):
#     for i in range(1):
#         classification = kNN(testingImages[:, i], trainingImages, trainingLabels, j)
#         # print(classification)
#         if classification == testLabels[i]:
#             accuracy += 1
#     accuracys.append(accuracy/testingImages.shape[1])
# print(accuracys)


# %%
# d = [5, 50, 100, 500]
# d=[5]
# trainData = []
# testData = []
# accuracys = []
# for i in range(len(d)):
#     trainData.append(pca(d[i]))
#     print(shape(pca(d[i])))
#     testData.append(pca(d[i]))
#     for j in k:
#         accuracy = 0
#         # for col in range(testData[i].shape[1]):
#         for col in range(10):
#             classification = kNN(testData[i][:, col], trainData[i], trainingLabels, j)
#             print(classification)
#             if classification == testLabels[col]:
#                 accuracy += 1
#         accuracys[i].append(accuracy/testingImages.shape[1])


print(pca(5))