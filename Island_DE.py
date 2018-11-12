import multiprocessing as mp
import random, time
from time import sleep
import numpy as np
import platform
import logging, os

# Define an output queue
output = mp.Queue()


def fobj(x):
    value = 0
    for i in range(len(x)):
        value += x[i] ** 2
    return value / len(x)


def de(fobj, bounds, mut=0.8, crossp=0.7, popsize=200, its=1000):
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


# Define a sample function
def rand_num(output):
    print('Process name {0} with id {1}'.format(mp.current_process().name, os.getpid()))
    num = random.random()
    output.put(num)


def cube(x):
    return x ** 3


def print_sysinfo():
    print('\nPython version  :', platform.python_version())
    print('compiler        :', platform.python_compiler())
    print('\nsystem     :', platform.system())
    print('release    :', platform.release())
    print('machine    :', platform.machine())
    print('processor  :', platform.processor())
    print('CPU count  :', mp.cpu_count())
    print('interpreter:', platform.architecture()[0])
    print('\n\n')


def func(arg):
    sleep(5)
    print("Hello, world! {}".format(arg))


def f(conn):
    conn.send(['hello world'])
    conn.close()


def main():
    # send it all to stderr
    mp.log_to_stderr()
    # get access to a logger and set its logging level to INFO
    logger = mp.get_logger()
    logger.setLevel(logging.INFO)
    # logger.warning('Error has occured')

    print_sysinfo()

    num_cpu = mp.cpu_count()
    # args = list(range(num_cpu))  # [0, 1, 2, 3, 4, 5, 6, 7]
    #
    # with mp.Pool(num_cpu) as p:
    #     res = p.map(func, args)  # function and its arguments

    # Setup a list of num_cpu processes that we want to run, and pass the arguments
    processes = [mp.Process(target=rand_num, args=(output,)) for x in range(num_cpu)]

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

    pool = mp.Pool(processes=4)

    # Pool.map and Pool.apply will lock the main program until all processes are finished
    pool_results = [pool.apply(cube, args=(x,)) for x in range(1, 7)]
    #    pool_results = [pool.map(cube, range(1,7))]

    #  the async variants will submit all processes at once and retrieve the results as soon as they are finished
    #     pool_results = [pool.apply_async(cube, args=(x,)) for x in range(1,7)]
    #     o = [p.get() for p in pool_results]
    #     print(o)
    print(pool_results)

    # use pipe to communicate between processes
    parent_conn, child_conn = mp.Pipe()
    proc = mp.Process(target=f, args=(child_conn,))
    proc.start()
    print(parent_conn.recv())
    proc.join()

    start_time = time.time()
    print("--- Testing DE ---")
    result = list(de(fobj, bounds=[(-100, 100)]*6, its=3000))
    print(result[-1])

    print("")
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
