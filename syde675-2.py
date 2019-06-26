import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.metrics import confusion_matrix
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA

#q2.a
fig = plt.figure(0)
ax = fig.add_subplot(111, aspect='equal')
p = [0.2, 0.3, 0.5]
mean = [[3, 2], [5, 4], [2, 5]]
cov = [[[1, -1], [-1, 2]], [[1, -1], [-1, 7]], [[0.5, 0.5], [0.5, 3]]]
sample = []
for i in range(3):
    sample.append(np.random.multivariate_normal(mean[i], cov[i], int(3000*p[i])))
x1, y1 = np.random.multivariate_normal(mean[0], cov[0], 600).T
x2, y2 = np.random.multivariate_normal(mean[1], cov[1], 900).T
x3, y3 = np.random.multivariate_normal(mean[2], cov[2], 1500).T
result = []
for i in range(3):
    result.append(np.random.multivariate_normal(mean[i], cov[i], int(3000*p[i])).T)

covx, covy, resultxx, resultxy, resultyx, resultyy, cov_after, val, vec, angle_x, angle_y, angle = [],\
    [], [], [], [], [], [], [], [], [], [], []
sample_pca = []


def angle_solution(i):
    covx.append(result[i][0, :]-np.mean(result[i][0, :]))
    covy.append(result[i][1, :]-np.mean(result[i][0, :]))
    resultxx.append(np.dot(covx[i], covx[i].T)/(3000*p[i]-1))
    resultxy.append(np.dot(covx[i], covy[i].T)/(3000*p[i]-1))
    resultyx.append(np.dot(covy[i], covx[i].T)/(3000*p[i]-1))
    resultyy.append(np.dot(covy[i], covy[i].T)/(3000*p[i]-1))
    cov_after.append(np.array([[resultxx[i], resultxy[i]], [resultyx[i], resultyy[i]]]))
    pca = PCA(n_components=2)
    sample_pca.append(pca.fit_transform(sample[i]))
    val.append(pca.explained_variance_)
    vec.append(pca.components_)
    angle_x.append(vec[i][0, 0])
    angle_y.append(vec[i][0, 1])
    angle.append(np.degrees(np.arctan2(angle_y[i], angle_x[i])))
    return val, angle[i]


def mean_solution(result):
    mean_x = np.mean(result[0, :])
    mean_y = np.mean(result[1, :])
    return plt.plot(mean_x, mean_y, 'k*')


val_aftersolution = []
for i in range(3):
    mean_solution(result[i])
    val_aftersolution, angle_aftersolution = angle_solution(i)
    b = Ellipse(xy=(np.mean(result[i][0, :]), np.mean(result[i][1, :])), width=2*np.sqrt(val_aftersolution[i][0]), height=2*np.sqrt(val_aftersolution[i][1]), angle=angle_aftersolution, fill=False, linewidth=2.5, edgecolor="black")
    ax.add_artist(b)


def ML_classifier(matrix):
    mean = [[3, 2], [5, 4], [2, 5]]
    cov = [[[1, -1], [-1, 2]], [[1, -1], [-1, 7]], [[0.5, 0.5], [0.5, 3]]]
    label = []
    for i in range(len(matrix)):
        mvn1 = multivariate_normal(mean[0], cov[0])
        boundary1 = mvn1.pdf(matrix[i])
        mvn2 = multivariate_normal(mean[1], cov[1])
        boundary2 = mvn2.pdf(matrix[i])
        mvn3 = multivariate_normal(mean[2], cov[2])
        boundary3 = mvn3.pdf(matrix[i])
        if ((boundary1 >= boundary2) and (boundary1 >= boundary3)):
            label.append(1)
        elif ((boundary2 >= boundary1) and (boundary2 >= boundary3)):
            label.append(2)
        elif ((boundary3 >= boundary2) and (boundary3 >= boundary1)):
            label.append(3)
    return np.array(label)


def MAP_classifier(matrix):
    mean = [[3, 2], [5, 4], [2, 5]]
    cov = [[[1, -1], [-1, 2]], [[1, -1], [-1, 7]], [[0.5, 0.5], [0.5, 3]]]
    label = []

    for i in range(len(matrix)):
        mvn1 = multivariate_normal(mean[0], cov[0])
        boundary1 = mvn1.pdf(matrix[i])*0.2
        mvn2 = multivariate_normal(mean[1], cov[1])
        boundary2 = mvn2.pdf(matrix[i])*0.3
        mvn3 = multivariate_normal(mean[2], cov[2])
        boundary3 = mvn3.pdf(matrix[i])*0.5
        if ((boundary1 >= boundary2) and (boundary1 >= boundary3)):
            label.append(1)
        elif ((boundary2 >= boundary1) and (boundary2 >= boundary3)):
            label.append(2)
        elif ((boundary3 >= boundary2) and (boundary3 >= boundary1)):
            label.append(3)
    return np.array(label)


def plot_decision_boundary(model, axis, style):
    X0, X1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 10)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 10)).reshape(-1, 1),
    )
    X_grid_matrix = np.c_[X0.ravel(), X1.ravel()]
    if (model == 'ML'):
        y_predict = ML_classifier(X_grid_matrix)
        y_predict_matrix = y_predict.reshape(X0.shape)
    if (model == 'MAP'):
        y_predict = MAP_classifier(X_grid_matrix)
        y_predict_matrix = y_predict.reshape(X0.shape)
    if (style == 2):
        plt.contour(X0, X1, y_predict_matrix, colors='orangered', linewidths=1.5, alpha=0.7)
    else:
        plt.contour(X0, X1, y_predict_matrix, colors='royalblue', linewidths=1.5, alpha=0.7)


plt.title('ML,MAP Decision Boundary')
plot_decision_boundary('ML', axis=[-2, 8, -5, 12], style=1)
plot_decision_boundary('MAP', axis=[-2, 8, -5, 12], style=2)
plt.savefig('q2a')
plt.show()

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, aspect='equal')
val_aftersolution1 = []
for i in range(3):
    mean_solution(result[i])
    val_aftersolution1, angle_aftersolution1 = angle_solution(i)
    b = Ellipse(xy=(np.mean(result[i][0, :]), np.mean(result[i][1, :])), width=2*np.sqrt(val_aftersolution1[i][0]), height=2*np.sqrt(val_aftersolution1[i][1]), angle=angle_aftersolution1, fill=False, linewidth=2.5, edgecolor="black")
    ax1.add_artist(b)
plt.title('ML')
plot_decision_boundary('ML', axis=[-2, 8, -5, 12], style=1)
plt.savefig('q2aML')
plt.show()

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111, aspect='equal')
val_aftersolution2 = []
for i in range(3):
    mean_solution(result[i])
    val_aftersolution2, angle_aftersolution2 = angle_solution(i)
    b = Ellipse(xy=(np.mean(result[i][0, :]), np.mean(result[i][1, :])), width=2*np.sqrt(val_aftersolution2[i][0]), height=2*np.sqrt(val_aftersolution2[i][1]), angle=angle_aftersolution2, fill=False, linewidth=2.5, edgecolor="black")
    ax2.add_artist(b)
plt.title('MAP')
plot_decision_boundary('MAP', axis=[-2, 8, -5, 12], style=2)
plt.savefig('q2aMAP')
plt.show()

#2.b
a = plt.subplot(111)
plt.scatter(x1, y1, c="yellow", s=1, alpha=0.5)
plt.scatter(x2, y2, c="deeppink", s=1, alpha=0.5)
plt.scatter(x3, y3, c="lime", s=1, alpha=0.5)
for i in range(3):
    # mean_solution(result[i])
    val_aftersolution, angle_aftersolution = angle_solution(i)
    b = Ellipse(xy=(np.mean(result[i][0, :]), np.mean(result[i][1, :])), width=2*np.sqrt(val_aftersolution[i][0]), height=2*np.sqrt(val_aftersolution[i][1]), angle=angle_aftersolution, fill=False, linewidth=1.5, edgecolor="black")
    b.set_edgecolor("black")
    a.add_artist(b)
plt.axis('equal')
label1 = np.ones((int(3000*p[0]), 1), int)
label2 = np.ones((int(3000*p[1]), 1), int)*2
label3 = np.ones((int(3000*p[2]), 1), int)*3
label_true = np.vstack((label1, label2, label3))
ms = np.vstack((sample[0], sample[1], sample[2]))
lable_predict_ML = ML_classifier(ms).T
lable_predict_MAP = MAP_classifier(ms).T
ML = confusion_matrix(label_true, lable_predict_ML)
MAP = confusion_matrix(label_true, lable_predict_MAP)
print(ML)
print(MAP)
error1 = round(((3000-ML[0][0]-ML[1][1]-ML[2][2])/3000), 3)
print('experimental P(ε) for ML_confusion: ', error1)
error2 = round(((3000-MAP[0][0]-MAP[1][1]-MAP[2][2])/3000), 3)
print('experimental P(ε) for MAP_confusion: ', error2)
plt.savefig('q2b')
plt.show()


