import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import whiten
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA


sample_pca = []
mean = [0, 0]
cov1, cov2 = [[1, 0], [0, 1]], [[1, 0.9], [0.9, 1]]
sample1 = np.random.multivariate_normal(mean, cov1, 1000)
sample2 = np.random.multivariate_normal(mean, cov2, 1000)
x1, y1 = np.random.multivariate_normal(mean, cov1, 1000).T
x2, y2 = np.random.multivariate_normal(mean, cov2, 1000).T


result1 = np.array([x1, y1])
cov1x = result1[0, :]-np.mean(result1[0, :])
cov1y = result1[1, :]-np.mean(result1[1, :])
result1xx = np.dot(cov1x, cov1x.T)/(1000-1)
result1xy = np.dot(cov1x, cov1y.T)/(1000-1)
result1yx = np.dot(cov1y, cov1x.T)/(1000-1)
result1yy = np.dot(cov1y, cov1y.T)/(1000-1)
cov1_after = np.array([[result1xx, result1xy], [result1yx, result1yy]])
print("cov1:", '\n', cov1_after)
val1, vec1 = np.linalg.eig(np.dot(cov1_after.T, cov1_after))
pca1 = PCA(n_components=2)
sample_pca.append(pca1.fit_transform(sample1))
val1_after = pca1.explained_variance_

angle1_x, angle1_y = vec1[:, 0]
angel1 = np.degrees(np.arctan2(angle1_y, angle1_x))
D1 = np.diag(1. / np.sqrt(val1+(1E-18)))
W1 = np.dot(np.dot(vec1, D1), vec1.T)
X_white1 = np.dot(result1.T, W1)

result1_afterwhiten = whiten(result1.T)

print("cov1_after_whiten:", '\n', result1_afterwhiten.T.dot(result1_afterwhiten)/1000)

result2 = np.array([x2, y2])
cov2x = result2[0, :]-np.mean(result2[0, :])
cov2y = result2[1, :]-np.mean(result2[1, :])
result2xx = np.dot(cov2x, cov2x.T)/(1000-1)
result2xy = np.dot(cov2x, cov2y.T)/(1000-1)
result2yx = np.dot(cov2y, cov2x.T)/(1000-1)
result2yy = np.dot(cov2y, cov2y.T)/(1000-1)
cov2_after = np.array([[result2xx, result2xy], [result2yx, result2yy]])
print("cov2:", '\n', cov2_after)

val2, vec2 = np.linalg.eig(np.dot(cov2_after, cov2_after.T))
pca2 = PCA(n_components=2)
sample_pca.append(pca2.fit_transform(sample2))
val2_after = pca2.explained_variance_
# angel2 = np.arccos(np.dot(vec2[0],np.array([1,0]))/(np.sqrt(vec2[0].dot(vec2[0]))))
angle2_x, angle2_y = vec2[:, 0]
angel2 = np.degrees(np.arctan2(angle2_y, angle2_x))
D2 = np.diag(1. / np.sqrt(val2+(1E-18)))
W2 = np.dot(np.dot(vec2, D2), vec2.T)
X_white2 = np.dot(result2.T, W2)
# print("cov2_after_whiten:", '\n', X_white2.T.dot(X_white2)/1000)

result2_afterwhiten = whiten(result2.T)
print("cov2_after_whiten:", '\n', result2_afterwhiten.T.dot(result2_afterwhiten)/1000)

plt.figure()
a = plt.subplot(111)
plt.title("Matrix a")
plt.xlim(-5, 5)
plt.xticks(np.arange(-4, 4, 1))
plt.plot(x1, y1, 'x', color="pink", zorder=0)
plt.savefig('1a')
plt.show()

plt.figure()
a = plt.subplot(111)
plt.title("Matrix a")
plt.xlim(-5, 5)
plt.xticks(np.arange(-4, 4, 1))
plt.plot(x1, y1, 'x', color="pink", zorder=0)
e = Ellipse(xy=(np.mean(result1[0, :]), np.mean(result1[1, :])), width=2*np.sqrt(val1_after[0]), height=2*np.sqrt(val1_after[1]), angle =angel1, fill=False)
e.set_edgecolor("purple")
a.add_artist(e)
plt.savefig('1b')
plt.show()


plt.figure()
b = plt.subplot(111)
plt.title("Matrix b")
plt.plot(x2, y2, 'x', color="pink", zorder=0)
plt.axis('equal')
plt.savefig('2a')
plt.show()


plt.figure()
b = plt.subplot(111)
plt.title("Matrix b")
plt.plot(x2, y2, 'x', color="pink", zorder=0)
f = Ellipse(xy=(np.mean(result2[0, :]), np.mean(result2[1, :])), width=2*np.sqrt(val2_after[1]), height=2*np.sqrt(val2_after[0]), angle =angel2, fill=False)
f.set_edgecolor("purple")
b.add_artist(f)
plt.axis('equal')
plt.savefig('2b')
plt.show()

