import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.metrics import confusion_matrix
import sklearn
from sklearn.naive_bayes import GaussianNB
import mlxtend
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import itertools
from scipy.stats import multivariate_normal
import random
import pandas as pd

#question2.a
sampleNo = 1000
sampleNo1 = 600
sampleNo2 = 900
sampleNo3 = 1500

fig = plt.figure(0)
ax = fig.add_subplot(111, aspect='equal')

#class1
cov1 = np.array([[1, -1], [-1, 2]])
mean1 = np.array([3,2])
# ms1 = np.random.multivariate_normal(mean1, cov1, sampleNo)
# plt.scatter(ms1[:,0],ms1[:,1],s = 2,alpha = .5)
#class2
cov2 = np.array([[1, -1], [-1, 7]])
mean2 = np.array([5,4])
# ms2 = np.random.multivariate_normal(mean2, cov2, sampleNo)
# ax = fig.add_subplot(111, aspect='equal')
# plt.scatter(ms2[:,0],ms2[:,1],s = 2,alpha = .5)
#class3
cov3 = np.array([[0.5, 0.5], [0.5, 3]])
mean3 = np.array([2,5])
# ms3 = np.random.multivariate_normal(mean3, cov3, sampleNo)
# ax = fig.add_subplot(111, aspect='equal')
# plt.scatter(ms3[:,0],ms3[:,1],s = 2,alpha = .5)
# plt.show()



def std_contour(mean, cov, position):
    #calculate the center of contour
    mean_x = mean[0]
    mean_y = mean[1]
    plt.plot(mean_x,mean_y,'r+')

    #calculate the direction of contour
    eigval, eigvec = np.linalg.eig(cov)
    idx = eigval.argsort()[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]
    long = -eigvec[:, 0]  # 确保特征向量方向和后文计算投影方向一致，故加负号
    short = eigvec[:, 1]
    tan1 = long[1]/long[0]
    angle1 = math.degrees(math.atan(tan1))

    #calculate the length of X & Y
    len_long = np.sqrt(eigval[0])
    len_short = np.sqrt(eigval[1])

    ##draw contour
    ax = fig.add_subplot(position, aspect='equal')
    ax.grid(True)
    e = Ellipse(xy=(mean_x, mean_y), width=len_long * 2,
                height=len_short * 2, angle=angle1,
                fill=False, linewidth=2.5, edgecolor="red")
    ax.add_artist(e)

std_contour(mean1, cov1, 111)
std_contour(mean2, cov2, 111)
std_contour(mean3, cov3, 111)

#decision boundary
def ML_classifier(matrix):
    mean1 = [3,2]
    mean2 = [5,4]
    mean3 = [2,5]
    cov1 = [[1, -1], [-1, 2]]
    cov2 = [[1, -1], [-1, 7]]
    cov3 = [[0.5, 0.5], [0.5, 3]]
    label = []

    for i in range(len(matrix)):
        x = matrix[i][0]
        y = matrix[i][1]
        mvn1 = multivariate_normal(mean1, cov1)
        likelihood1 = mvn1.pdf(matrix[i])
        mvn2 = multivariate_normal(mean2, cov2)
        likelihood2 = mvn2.pdf(matrix[i])
        mvn3 = multivariate_normal(mean3, cov3)
        likelihood3 = mvn3.pdf(matrix[i])

        if ((likelihood1 >= likelihood2) and (likelihood1 >= likelihood3)):
            label.append(1)
        elif ((likelihood2 >= likelihood1) and (likelihood2 >= likelihood3)):
            label.append(2)
        elif ((likelihood3 >= likelihood2) and (likelihood3 >= likelihood1)):
            label.append(3)
        #likelihood1 = np.sqrt(2*np.pi*cov1_x*cov1_y*np.sqrt(1-cov1_xy**2))

    return np.array(label)

def MAP_classifier(matrix):
    mean1 = [3,2]
    mean2 = [5,4]
    mean3 = [2,5]
    cov1 = [[1, -1], [-1, 2]]
    cov2 = [[1, -1], [-1, 7]]
    cov3 = [[0.5, 0.5], [0.5, 3]]
    label = []

    for i in range(len(matrix)):
        mvn1 = multivariate_normal(mean1, cov1)
        likelihood1 = mvn1.pdf(matrix[i])*0.2
        mvn2 = multivariate_normal(mean2, cov2)
        likelihood2 = mvn2.pdf(matrix[i])*0.3
        mvn3 = multivariate_normal(mean3, cov3)
        likelihood3 = mvn3.pdf(matrix[i])*0.5
        if ((likelihood1 >= likelihood2) and (likelihood1 >= likelihood3)):
            label.append(1)
        elif ((likelihood2 >= likelihood1) and (likelihood2 >= likelihood3)):
            label.append(2)
        elif ((likelihood3 >= likelihood2) and (likelihood3 >= likelihood1)):
            label.append(3)
        #likelihood1 = np.sqrt(2*np.pi*cov1_x*cov1_y*np.sqrt(1-cov1_xy**2))

    return np.array(label)


def plot_decision_boundary(model, axis, style):
    # meshgrid函数用两个坐标轴上的点在平面上画格，返回坐标矩阵
    X0, X1 = np.meshgrid(
        # 随机两组数，起始值和密度由坐标轴的起始值决定
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 10)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 10)).reshape(-1, 1),
    )
    # ravel()方法将高维数组降为一维数组，c_[]将两个数组以列的形式拼接起来，形成矩阵
    X_grid_matrix = np.c_[X0.ravel(), X1.ravel()]

    # 通过训练好的逻辑回归模型，预测平面上这些点的分类
    if (model == 'ML'):
        y_predict = ML_classifier(X_grid_matrix)
        y_predict_matrix = y_predict.reshape(X0.shape)
    if (model == 'MAP'):
        y_predict = MAP_classifier(X_grid_matrix)
        y_predict_matrix = y_predict.reshape(X0.shape)
    # else :
    #     y_predict = model.predict(X_grid_matrix)
    #     y_predict_matrix = y_predict.reshape(X0.shape)

    # 设置色彩表
    from matplotlib.colors import ListedColormap
    my_colormap1 = ListedColormap(['lightcoral', 'mistyrose', 'pink'])
    my_colormap2 = ListedColormap(['deepskyblue', 'dodgerblue', 'skyblue'])

    # 绘制等高线，并且填充等高区域的颜色
    if (style == 2):
        #plt.contourf(X0, X1, y_predict_matrix, cmap=my_colormap2, alpha=0.8)
        plt.contour(X0, X1, y_predict_matrix, colors='deepskyblue', linewidths=1.5, alpha=0.7)
    else:
        # plt.contourf(X0, X1, y_predict_matrix, cmap=my_colormap1, alpha=0.8)
        plt.contour(X0, X1, y_predict_matrix, colors='firebrick', linewidths=1.5, alpha=0.7)

plot_decision_boundary('ML', axis=[-2, 8, -6, 13], style=1)
plot_decision_boundary('MAP', axis=[-2, 8, -6, 13], style=2)
plt.show()


#question2.b
#class1
cov1 = np.array([[1, -1], [-1, 2]])
mean1 = np.array([3,2])
ms1 = np.random.multivariate_normal(mean1, cov1, sampleNo1)
plt.scatter(ms1[:,0],ms1[:,1],s = 2,alpha = .5)
#class2
cov2 = np.array([[1, -1], [-1, 7]])
mean2 = np.array([5,4])
ms2 = np.random.multivariate_normal(mean2, cov2, sampleNo2)
plt.scatter(ms2[:,0],ms2[:,1],s = 2,alpha = .5)
#class3
cov3 = np.array([[0.5, 0.5], [0.5, 3]])
mean3 = np.array([2,5])
ms3 = np.random.multivariate_normal(mean3, cov3, sampleNo3)
plt.scatter(ms3[:,0],ms3[:,1],s = 2,alpha = .5)


label1 = np.ones((sampleNo1,1),int)
label2 = np.ones((sampleNo2,1),int)*2
label3 = np.ones((sampleNo3,1),int)*3
label_true = np.vstack((label1,label2,label3))
ms = np.vstack((ms1,ms2,ms3))
fig = plt.figure(0)
ax = fig.add_subplot(111, aspect='equal')
plt.scatter(ms[:,0],ms[:,1],s = 2,alpha = .5)
lable_predict_ML = ML_classifier(ms).T
lable_predict_MAP = MAP_classifier(ms).T

ML_confusion = confusion_matrix(label_true, lable_predict_ML)
MAP_confusion = confusion_matrix(label_true, lable_predict_MAP)
print(ML_confusion)
print(MAP_confusion)
# experimental P(ε)
def calculate_error(matrix):
    sum = matrix[0][0]+matrix[1][1]+matrix[2][2]
    return (3000-sum)/3000
print(calculate_error(ML_confusion))
print(calculate_error(MAP_confusion))
plt.show()




# lable1 = np.ones((sampleNo,1),int)
# ms1 = np.hstack((ms1,lable1))
# lable2 = np.ones((sampleNo,1),int)*2
# ms2 = np.hstack((ms2,lable2))
# lable3 = np.ones((sampleNo,1),int)*3
# ms3 = np.hstack((ms3,lable3))
# ms_MLE = np.vstack((ms1,ms2,ms3))
#
# ms1 = ms1[0:sampleNo1]
# ms2 = ms2[0:sampleNo2]
# ms3 = ms3[0:sampleNo3]
# ms_MAP = np.vstack((ms1,ms2,ms3))
#
# X = ms_MLE[:,[0,1]]
# y = ms_MLE[:,2].astype(np.integer)
# clf = GaussianNB() #create classifier object
# clf.fit(X, y)  #Fitting the classifier model with training data
# #fig = plot_decision_regions(X=X, y=y, clf=clf)
# plot_decision_boundary(clf, axis=[-2, 8, -6, 13], style=1)
#
# X = ms_MAP[:,[0,1]]
# y = ms_MAP[:,2].astype(np.integer)
# clf = GaussianNB() #create classifier object
# clf.fit(X, y)  #Fitting the classifier model with training data
# #fig = plot_decision_regions(X=X, y=y, clf=clf)
# plot_decision_boundary(clf, axis=[-2, 8, -6, 13],style=2)
# plt.show()



# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Ellipse
# from sklearn.decomposition import PCA
#
# p = [0.2, 0.3, 0.5]
# mean = [[3, 2], [5, 4], [2, 5]]
# cov = [[[1, -1], [-1, 2]], [[1, -1], [-1, 7]], [[0.5, 0.5], [0.5, 3]]]
# sample = []
# for i in range(3):
#     sample.append(np.random.multivariate_normal(mean[i], cov[i], int(3000*p[i])))
# x1, y1 = np.random.multivariate_normal(mean[0], cov[0], 600).T
# x2, y2 = np.random.multivariate_normal(mean[1], cov[1], 900).T
# x3, y3 = np.random.multivariate_normal(mean[2], cov[2], 1500).T
#
#
# result = []
# # print(type(result))
# for i in range(3):
#     result.append(np.random.multivariate_normal(mean[i], cov[i], int(3000*p[i])).T)
#
# covx, covy, resultxx, resultxy, resultyx, resultyy, cov_after, val, vec, angle_x, angle_y, angle = [],\
#     [], [], [], [], [], [], [], [], [], [], []
# sample_pca = []
#
#
# def angle_solution(i):
#     covx.append(result[i][0, :]-np.mean(result[i][0, :]))
#     covy.append(result[i][1, :]-np.mean(result[i][0, :]))
#     resultxx.append(np.dot(covx[i], covx[i].T)/(3000*p[i]-1))
#     resultxy.append(np.dot(covx[i], covy[i].T)/(3000*p[i]-1))
#     resultyx.append(np.dot(covy[i], covx[i].T)/(3000*p[i]-1))
#     resultyy.append(np.dot(covy[i], covy[i].T)/(3000*p[i]-1))
#     cov_after.append(np.array([[resultxx[i], resultxy[i]], [resultyx[i], resultyy[i]]]))
#     pca = PCA(n_components=2)
#     sample_pca.append(pca.fit_transform(sample[i]))
#     val.append(pca.explained_variance_)
#     vec.append(pca.components_)
#     angle_x.append(vec[i][0, 0])
#     angle_y.append(vec[i][0, 1])
#     angle.append(np.degrees(np.arctan2(angle_y[i], angle_x[i])))
#     return val, angle[i]
#
#
# def mean_solution(result):
#     mean_x = np.mean(result[0, :])
#     mean_y = np.mean(result[1, :])
#     print(mean_x, mean_y)
#     return plt.scatter(mean_x, mean_y, c='black', s=0.2)
# plt.figure()
# a = plt.subplot(111)
# plt.scatter(x1, y1, c="red", s=0.35)
# plt.scatter(x2, y2, c="blue", s=0.35)
# plt.scatter(x3, y3, c="green", s=0.35)
#
# val_aftersolution = []
#
# for i in range(3):
#     mean_solution(result[i])
#     val_aftersolution, angle_aftersolution = angle_solution(i)
#     b = Ellipse(xy=(np.mean(result[i][0, :]), np.mean(result[i][1, :])), width=2*np.sqrt(val_aftersolution[i][0]), height=2*np.sqrt(val_aftersolution[i][1]), angle=angle_aftersolution, fill=False)
#     b.set_edgecolor("black")
#     a.add_artist(b)
# plt.axis('equal')
# plt.show()