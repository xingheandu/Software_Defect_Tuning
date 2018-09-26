from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def de(func, bounds, mut=0.8, crossp=0.7, popsize=20, its=200):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)

    # pdb.set_trace()
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff

    # convert from float to integer
    pop_denorm_convert = pop_denorm.tolist()

    result_list = []
    temp_list = []

    for index in pop_denorm_convert:
        temp_list.append(np.int_(np.round_(index[0])))
        temp_list.append(np.int_(np.round_(index[1])))
        temp_list.append(np.int_(np.round_(index[2])))
        temp_list.append(np.int_(np.round_(index[3])))
        temp_list.append(float('%.2f' % index[4]))
        temp_list.append(np.int(np.round_(index[5])))
        result_list.append(temp_list)
        temp_list = []

    fitness = np.asarray([func(index[0], index[1], index[2], index[3], index[4], index[5])
                          for index in result_list])

    best_idx = np.argmax(fitness)
    best = pop_denorm[best_idx]

    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = a + mut * (b - c)
            for i, v in enumerate(mutant):
                if 0 < v < 1: continue
                if v < 0: mutant[i] = v + 1
                if v > 1: mutant[i] = v - 1
            # mutant = np.clip(res, 0, 1)

            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True

            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            trail_denorm_convert = trial_denorm.tolist()
            f = func(np.int_(np.round_(trail_denorm_convert[0])), np.int_(np.round_(trail_denorm_convert[1])), np.int_(np.round_(trail_denorm_convert[2])),
                     np.int_(np.round_(trail_denorm_convert[3])), float('%.2f' % trail_denorm_convert[4]), np.int_(np.round_(trail_denorm_convert[5])))

            if f > fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f > fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]


features = pd.read_csv(r'C:\Users\terry\PycharmProjects\tutorial\Optimizer\cm1.csv')
features = features.sample(frac=1)
X = features.iloc[:, :-1]
y = features['defects'].map({True: 1, False: 0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)


# def rf(n_estimators, min_samples_leaf, min_samples_split, max_leaf_nodes, max_features, max_depth):
#     forest = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
#                                     max_leaf_nodes=max_leaf_nodes, max_features=max_features, max_depth=max_depth)
#
#     forest.fit(X_train, y_train)
#     y_pred = forest.predict(X_test)
#     precision = precision_score(y_test, y_pred, average="macro")
#     print(precision)
#
#     return precision
#
#
# result = list(de(rf, bounds=[(50, 150), (1, 20), (2, 20), (2, 50), (0.01, 1), (1, 10)]))
# print(result[-1])


forest = RandomForestClassifier(n_estimators=100, min_samples_leaf=1, min_samples_split=2, max_leaf_nodes=None,
                                max_features=None, max_depth=5)

forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
# print(y_pred)

print(f1_score(y_test, y_pred, average="macro"))
print(precision_score(y_test, y_pred, average="macro"))
print(recall_score(y_test, y_pred, average="macro"))
#

# default value result
# 0.46120689655172414
# 0.4385245901639344
# 0.4863636363636364

