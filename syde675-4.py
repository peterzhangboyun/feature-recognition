from numpy import *
import numpy as np
import struct

np.seterr(invalid='ignore')


def load_images(file_name):
    binfile = open(file_name, 'rb')
    buffers = binfile.read()
    magic, num, rows, cols = struct.unpack_from('>IIII',buffers, 0)
    bits = num * rows * cols
    images = struct.unpack_from('>' + str(bits) + 'B', buffers, struct.calcsize('>IIII'))
    binfile.close()
    images = np.reshape(images, [num, rows * cols])
    images = images.T
    return images


def load_labels(file_name):
    binfile = open(file_name, 'rb')
    buffers = binfile.read()
    magic, num = struct.unpack_from('>II', buffers, 0)
    labels = struct.unpack_from('>' + str(num) + "B", buffers, struct.calcsize('>II'))
    binfile.close()
    labels = np.reshape(labels, [num])
    return labels


global data
data = load_images('t10k-images.idx3-ubyte')
data = data.T


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
    return finalData


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


def kNN1(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[1]
    print('KNN first', numSamples)
    diff = np.tile(newInput.reshape(1, -1), (numSamples, 1)).T - dataSet
    diff=np.array(diff)
    squaredDiff = np.power(diff, 2)
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


testingImages = load_images('t10k-images.idx3-ubyte')
testingImages = np.float32(testingImages)
testLabels = load_labels('t10k-labels.idx1-ubyte')
testLabels = np.float32(testLabels)
trainingImages = load_images('train-images.idx3-ubyte')
trainingImages = np.float32(trainingImages)
trainingLabels = load_labels('train-labels.idx1-ubyte')
trainingLabels = np.float32(trainingLabels)

# Q4.a
k = [1, 3, 5, 11]
accuracys_knn = []
for j in k:
    accuracy = 0
    for i in range(testingImages.shape[1]):
        classification = kNN(testingImages[:, i], trainingImages, trainingLabels, j)
        if classification == testLabels[i]:
            accuracy += 1
    accuracys_knn.append(accuracy/testingImages.shape[1])
print(accuracys_knn)


# Q4.b
d = [5, 50, 100, 500]
trainData = []
testData = []
accuracys_pca_knn = []
for i in range(len(d)):
    trainData.append(pca(d[i]))
    print(shape(pca(d[i])))
    testData.append(pca(d[i]))
    for j in k:
        accuracy = 0
        for col in range(testData[i].shape[1]):
            a = testData[i].T
            classification = kNN1(a[:, col], trainData[i].T, trainingLabels, j)
            if classification == testLabels[col]:
                accuracy += 1
        accuracys_pca_knn.append(accuracy/testData[i].shape[1])

print(accuracys_pca_knn)
