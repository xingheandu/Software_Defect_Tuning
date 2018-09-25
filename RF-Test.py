from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cross_validation import train_test_split
import numpy as np
from scipy.io import arff
from matplotlib.colors import ListedColormap

# data = arff.loadarff(r'C:\Users\Terry\PycharmProjects\MLOptimization\Optimizer\cm1.arff')
# df = pd.DataFrame(data[0])
#
# print(df.head())
#

def plot_decision_regions(X, y, classifier,
                          test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='',
                    alpha=1.0, linewidth=1, marker='o',
                    s=55, label='test set')


"""Random forests creates decision trees on randomly selected
data samples, gets prediction from each tree and selects the 
best solution by means of voting"""

"""In a classification problem, each tree votes and the most
popular class is chosen as the final result. 
In the case of regression, the average of all the tree outputs
is considered as the final result."""

# iris = datasets.load_iris()
# X = iris.data[:,[2,3]]
# y = iris.target
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#
# forest = RandomForestClassifier(criterion='entropy',
#                                 n_estimators=10,
#                                 random_state=1,
#                                 n_jobs=2)
#
# forest.fit(X_train, y_train)
#
# X_combined = np.vstack((X_train, X_test))
# y_combined = np.hstack((y_train, y_test))
# plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105,150))
# plt.xlabel('petal length')
# plt.ylabel('petal width')
# plt.legend(loc='upper left')
# plt.show()


features = arff.loadarff(r'C:\Users\Terry\PycharmProjects\MLOptimization\Optimizer\cm1.arff')
# features = pd.get_dummies(features)
features = np.array(features)
print(features)
labels = np.array(features['defects'])
print(labels)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25, random_state=42)

forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=10,
                                random_state=1,
                                n_jobs=2)

forest.fit(train_features, train_labels)

# X_combined = np.vstack((train_features, test_features))
# y_combined = np.hstack((train_labels, test_labels))
# plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105,150))
# # plt.xlabel('petal length')
# # plt.ylabel('petal width')
# # plt.legend(loc='upper left')
# plt.show()
