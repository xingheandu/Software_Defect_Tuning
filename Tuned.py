import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
import warnings
import sklearn.metrics as metrics
import time

warnings.filterwarnings('ignore')

DATASET_PATH = r"C:\Users\Terry\Documents\Software_Defect_Tuning\testDataset\xerces-1.3.csv"


def read_data(path):
    data = pd.read_csv(path, skipinitialspace=True)
    data = data.iloc[:, 3:]
    return data


def data_statistics(dataset):
    # print the first 5 rows of data
    print(dataset.head(5))
    print('The shape of our dataset is:', dataset.shape)
    # Descriptive statistics for each column
    print(dataset.describe())


def split_dataset(dataset, train_percentage):
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1].map(lambda x: 'False' if x == 0 else 'True')
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=train_percentage, random_state=42)
    return train_x, test_x, train_y, test_y


def de_rf(func, bounds, mut=0.8, crossp=0.7, popsize=60, its=200):
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
            f = func(np.int_(np.round_(trail_denorm_convert[0])), np.int_(np.round_(trail_denorm_convert[1])),
                     np.int_(np.round_(trail_denorm_convert[2])),
                     np.int_(np.round_(trail_denorm_convert[3])), float('%.2f' % trail_denorm_convert[4]),
                     np.int_(np.round_(trail_denorm_convert[5])))

            if f > fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f > fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]


def de_mlpn(func, bounds, mut=0.8, crossp=0.7, popsize=60, its=200):
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
        temp_list.append(index[0])
        temp_list.append(index[1])
        temp_list.append(index[2])
        temp_list.append(np.int(np.round_(index[3])))
        temp_list.append(index[4])
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
            f = func(trail_denorm_convert[0], trail_denorm_convert[1],
                     trail_denorm_convert[2],
                     np.int(np.round_(trail_denorm_convert[3])), trail_denorm_convert[4],
                     np.int(np.round_(trail_denorm_convert[5])))

            if f > fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f > fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]


def random_forest(features, target):
    # rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
    #                             max_depth=9, max_features=0.8, max_leaf_nodes=23,
    #                             min_impurity_decrease=0.0, min_impurity_split=None,
    #                             min_samples_leaf=3, min_samples_split=4,
    #                             min_weight_fraction_leaf=0.0, n_estimators=108, n_jobs=None,
    #                             oob_score=False, random_state=0, verbose=0, warm_start=False)
    rf = RandomForestClassifier()
    rf.fit(features, target)
    return rf


def rf_tuning(n_estimators, min_samples_leaf, min_samples_split, max_leaf_nodes, max_features, max_depth):
    """
    Define the tuning target function, e.g., to achieve higher precision score in RandomForest
    """
    rf_tuning = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                       min_samples_split=min_samples_split, max_leaf_nodes=max_leaf_nodes,
                                       max_features=max_features, max_depth=max_depth)
    rf_tuning.fit(train_x, train_y)
    predictions = rf_tuning.predict(test_x)
    recall = recall_score(test_y, predictions, average="macro")
    return recall


def multilayer_perceptron(features, target):
    # mlpn = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
    #                      beta_1=0.9, beta_2=0.999, early_stopping=False,
    #                      epsilon=1e-08, hidden_layer_sizes=(5, 2),
    #                      learning_rate='constant', learning_rate_init=0.001,
    #                      max_iter=200, momentum=0.9, n_iter_no_change=10,
    #                      nesterovs_momentum=True, power_t=0.5, random_state=1,
    #                      shuffle=True, solver='lbfgs', tol=0.0001,
    #                      validation_fraction=0.1, verbose=False, warm_start=False)
    mlpn = MLPClassifier()
    mlpn.fit(features, target)
    return mlpn


def mlpn_tuning(alpha, learning_rate_init, power_t, max_iter, momentum, n_iter_no_change):
    mlpn = MLPClassifier(alpha=alpha, learning_rate_init=learning_rate_init, power_t=power_t, max_iter=max_iter,
                         momentum=momentum, n_iter_no_change=n_iter_no_change, solver="sgd")
    mlpn.fit(train_x, train_y)
    predictions = mlpn.predict(test_x)
    recall = recall_score(test_y, predictions, average="macro")
    return recall


def result_statistics(predictions):
    print("Precision  :: ", precision_score(test_y, predictions, average='macro'))
    print("recall score ::", recall_score(test_y, predictions, average='macro'))
    print("f1 score :: ", f1_score(test_y, predictions, average='macro'))

    # precision, recall, fscore, support = score(test_y, predictions)
    # print('precision: {}'.format(precision))
    # print('recall: {}'.format(recall))
    # print('fscore: {}'.format(fscore))
    # print('support: {}'.format(support))


def main():
    dataset = read_data(DATASET_PATH)

    # Get basic statistics of the loaded dataset
    # data_statistics(dataset)

    global train_x, test_x, train_y, test_y

    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.25)
    # Train and Test dataset size details
    # print("Train_x Shape :: ", train_x.shape)
    # print("Train_y Shape :: ", train_y.shape)
    # print("Test_x Shape :: ", test_x.shape)
    # print("Test_y Shape :: ", test_y.shape)

    # print("")
    # print("----------Random Forest----------")
    # rf = random_forest(train_x, train_y)
    # print("Trained model:", rf)
    #
    # rf_predictions = rf.predict(test_x)
    # # print("Train Accuracy :: ", accuracy_score(train_y, rf.predict(train_x)))
    # result_statistics(rf_predictions)

    start_time_rf_de = time.time()
    print("")
    print("----------Tuning Random Forest----------")
    de_rf_result = list(de_rf(rf_tuning, bounds=[(10, 150), (1, 20), (2, 20), (2, 50), (0.01, 1), (1, 10)]))
    print(de_rf_result[-1])

    # print("")
    # print("----------Multilayer Perceptron----------")
    # mlpn = multilayer_perceptron(train_x, train_y)
    # print("Trained model:", mlpn)
    #
    # mlpn_predictions = mlpn.predict(test_x)
    # # print("Train Accuracy :: ", accuracy_score(train_y, mlpn.predict(train_x)))
    # result_statistics(mlpn_predictions)
    #

    print("")
    print("--- %s seconds ---" % (time.time() - start_time_rf_de))

    print("")
    start_time_mlpn_de = time.time()
    print("----------Tuning Multilayer Perceptron----------")
    de_mlpn_result = list(
        de_mlpn(mlpn_tuning, bounds=[(0.0001, 0.001), (0.001, 0.01), (0.1, 1), (50, 300), (0.1, 1), (10, 100)]))
    print(de_mlpn_result[-1])

    print("")
    print("--- %s seconds ---" % (time.time() - start_time_mlpn_de))


if __name__ == "__main__":
    main()
