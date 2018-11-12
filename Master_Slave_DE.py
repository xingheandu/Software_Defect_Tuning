import multiprocessing as mp
import random, time
from time import sleep
import numpy as np
import platform
import logging, os

# Define an output queue
output = mp.Queue()

global bounds, popsize, dimensions, pop, min_b, max_b, diff, pop_denorm, fitness, best, best_idx, mut, crossp, its


def fobj(x):
    value = 0
    for i in range(len(x)):
        value += x[i] ** 2
    return value / len(x)


def de_initialization(fobj, bounds, popsize=200):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    return dimensions, pop, min_b, diff, best, best_idx, fitness


def de_innerloop(output, its, popsize, pop, mut, dimensions, crossp, min_b, diff, lock, fitness, best_idx, best):
    for i in range(its // mp.cpu_count()):
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
            lock.acquire()
            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
            lock.release()
        # yield best, fitness[best_idx]
    output.put("best: {0}, fitness[best_idx]: {1} ".format(best, fitness[best_idx]))


def de_sequence(fobj, bounds, mut=0.8, crossp=0.7, popsize=200, its=3000):
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


def main():
    # send it all to stderr
    mp.log_to_stderr()
    # get access to a logger and set its logging level to INFO
    logger = mp.get_logger()
    logger.setLevel(logging.INFO)

    print("--- Testing Sequence DE ---")
    start_time_seq = time.time()
    result_seq = list(de_sequence(fobj, bounds=[(-100, 100)] * 6))
    print(result_seq[-1])
    print("")
    print("--- %s seconds ---" % (time.time() - start_time_seq))

    sleep(5)

    start_time_para = time.time()
    print("--- Testing Parallel DE ---")

    # result_para = list(de_parallel(fobj, bounds=[(-100, 100)] * 6))
    # print(result_para[-1])

    # initialization
    bounds = [(-100, 100)] * 6
    popsize = 200
    mut = 0.8
    crossp = 0.7
    popsize = 200
    its = 3000

    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
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
            output, its, popsize, pop, mut, dimensions, crossp, min_b, diff, lock, fitness, best_idx, best)))

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
    print("--- %s seconds ---" % (time.time() - start_time_para))


if __name__ == "__main__":
    main()
