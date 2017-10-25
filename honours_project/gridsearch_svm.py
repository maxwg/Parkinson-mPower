from sklearn.svm import SVC
from thread_helper import processIterableInThreads
from queue import Queue
import numpy as np
def gridsearch_svm(eval_fn, max_dim, search_params = None):
    """
    :param eval_fn: (model -> cross validation results)
    :param max_dim: cross validation dimension to maximise
    :param [search_params]: params for search
    :return:
    """
    if search_params is None:
        search_params = {'kernel': ['rbf'],
                         'gamma': [1e-6, 1e-5, 1e-4, 1e-3],
                         'C': [100, 1000, 2000, 5000, 10000]}

    results = Queue()
    combos = []
    for kernel in search_params["kernel"]:
        for C in search_params['C']:
            if kernel == 'linear':
                combos.append((kernel, C, 'auto'))
            else:
                for gamma in search_params["gamma"]:
                    combos.append((kernel, C, gamma))

    def evaluate_SVM(combos, thread_idx):
        for combo in combos:
            res = eval_fn(lambda: SVC(kernel=combo[0], C=combo[1], gamma=combo[2], probability=True))
            print(np.mean(res, axis=0), (combo))
            results.put((res[max_dim], (combo)))

    processIterableInThreads(combos, evaluate_SVM, 4)
    return list(results.queue)