import multiprocessing as mp
import pandas as pd
import random, time
from time import sleep
import numpy as np
import platform
import logging, os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# Define an output queue
output = mp.Queue()
DATASET_PATH = r"/home/rshu/Software_Defect_Tuning/testDataset/prop-5.csv"

global bounds, popsize, dimensions, pop, min_b, max_b, diff, pop_denorm, fitness, best, best_idx, mut, crossp, its


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


# def fobj(x):
#     value = 0
#     for i in range(len(x)):
#         value += x[i] ** 2
#     return value / len(x)


def de_initialization(fobj, bounds, popsize=60):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    return dimensions, pop, min_b, diff, best, best_idx, fitness


def de_innerloop(output, its, popsize, pop, mut, dimensions, crossp, min_b, diff, lock, fitness, best_idx, best,
                 train_x, test_x, train_y, test_y):
    for i in range(its // mp.cpu_count()):
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
            f = mlpn_tuning(trail_denorm_convert[0], trail_denorm_convert[1],
                            trail_denorm_convert[2],
                            np.int(np.round_(trail_denorm_convert[3])), trail_denorm_convert[4],
                            np.int(np.round_(trail_denorm_convert[5])), train_x, test_x, train_y, test_y)

            lock.acquire()
            # f = fobj(trial_denorm)
            if f > fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f > fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
            lock.release()
        # yield best, fitness[best_idx]
    output.put("best: {0}, fitness[best_idx]: {1} ".format(best, fitness[best_idx]))


def de_sequence(fobj, bounds, mut=0.8, crossp=0.9, popsize=60, its=3000):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]

    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]


def rf_tuning(n_estimators, min_samples_leaf, min_samples_split, max_leaf_nodes, max_features, max_depth, train_x,
              test_x, train_y, test_y):
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


def mlpn_tuning(alpha, learning_rate_init, power_t, max_iter, momentum, n_iter_no_change, train_x, test_x, train_y,
                test_y):
    mlpn = MLPClassifier(alpha=alpha, learning_rate_init=learning_rate_init, power_t=power_t, max_iter=max_iter,
                         momentum=momentum, n_iter_no_change=n_iter_no_change, solver="sgd")
    mlpn.fit(train_x, train_y)
    predictions = mlpn.predict(test_x)
    recall = recall_score(test_y, predictions, average="macro")
    return recall


def main():
    # send it all to stderr
    mp.log_to_stderr()
    # get access to a logger and set its logging level to INFO
    logger = mp.get_logger()
    logger.setLevel(logging.INFO)

    dataset = read_data(DATASET_PATH)
    # global train_x, test_x, train_y, test_y

    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.25)

    # print("--- Testing Sequence DE ---")
    # start_time_seq = time.time()
    # result_seq = list(de_sequence(fobj, bounds=[(-100, 100)] * 6))
    # print(result_seq[-1])
    # print("")
    # print("--- %s seconds ---" % (time.time() - start_time_seq))
    #
    # sleep(5)

    print("--- Tuning Multilayer Perceptron with Parallel DE ---")
    start_time_mlpn_tuning_para = time.time()

    # result_para = list(de_parallel(fobj, bounds=[(-100, 100)] * 6))
    # print(result_para[-1])

    # initialization
    bounds = [(0.0001, 0.001), (0.001, 0.01), (0.1, 1), (50, 300), (0.1, 1), (10, 100)]
    mut = 0.8
    crossp = 0.7
    popsize = 60
    its = 100

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

    fitness = np.asarray(
        [mlpn_tuning(index[0], index[1], index[2], index[3], index[4], index[5], train_x, test_x, train_y, test_y)
         for index in result_list])

    best_idx = np.argmax(fitness)
    best = pop_denorm[best_idx]

    print("Dimension:", dimensions)
    print("pop:", pop)
    print("min_b:", min_b)
    print("max_b:", max_b)
    print("diff:", diff)
    print("pop_denorm:", pop_denorm)
    print("fitness:", fitness)
    print("best_idx:", best_idx)
    print("best:", best)

    lock = mp.Lock()
    # execute loops in each process
    processes = []
    for x in range(mp.cpu_count()):
        processes.append(mp.Process(target=de_innerloop, args=(
            output, its, popsize, pop, mut, dimensions, crossp, min_b, diff, lock, fitness, best_idx, best, train_x,
            test_x, train_y, test_y)))

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    # Without join() function call, process will remain idle and won’t terminate
    for p in processes:
        p.join()

    # Get process results from the output queue
    results = [output.get() for p in processes]
    print(results)

    print("")
    print("--- %s seconds ---" % (time.time() - start_time_mlpn_tuning_para))
    print("")


if __name__ == "__main__":
    main()
