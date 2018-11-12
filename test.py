import time
import threading


def t():
    with open('/dev/urandom', 'rb') as f:
        for x in range(100):
            f.read(4 * 65535)


if __name__ == '__main__':
    start_time = time.time()
    t()
    t()
    t()
    t()
    print("Sequential run time: %.2f seconds" % (time.time() - start_time))

    start_time = time.time()
    t1 = threading.Thread(target=t)
    t2 = threading.Thread(target=t)
    t3 = threading.Thread(target=t)
    t4 = threading.Thread(target=t)
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t1.join()
    t2.join()
    t3.join()
    t4.join()
    print("Parallel run time: %.2f seconds" % (time.time() - start_time))
