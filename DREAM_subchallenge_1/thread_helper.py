import threading
from multiprocessing import Process, Queue, Manager
import math
from itertools import islice


class AnonThread(threading.Thread):
    def __init__(self, target, *args):
        threading.Thread.__init__(self)
        self._target = target
        self._args = args

    def run(self):
        self._target(*self._args)

def chunkIterableDict(data, size=8):
    it = iter(data)
    for i in range(0, len(data), math.ceil(len(data) / size)):
        yield {k:data[k] for k in islice(it, math.ceil(len(data) / size))}

def chunkIterableList(seq, chunks):
    size = math.ceil(len(seq) / chunks)
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def processIterableInThreads(iterable, fn, t_count):
    threads = []
    chunkFn = chunkIterableDict if isinstance(iterable, (dict)) else chunkIterableList
    for t_num, group in enumerate(chunkFn(iterable, t_count)):
        threads.append(AnonThread(fn, group, t_num))
    [t.start() for t in threads]
    [t.join() for t in threads]

def processIterableInProcesses(iterable, fn, t_count):
    procs = []
    manager = Manager()
    q = manager.Queue()
    chunkFn = chunkIterableDict if isinstance(iterable, (dict)) else chunkIterableList
    chunkFn(iterable, t_count)
    for t_num, group in enumerate(chunkFn(iterable, t_count)):
        procs.append(Process(target=fn, args=(group, t_num, q)))
    [t.start() for t in procs]
    [t.join() for t in procs]
    return q
