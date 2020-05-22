import time
import logging
import torch

def timeit(func):
    def timer(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        logger.info('{} in {} secs'. format(func.__name__, time.time() - start))
        return res
    return timer


LOG_LEVEL = logging.INFO
SAVE_FILE_LOG = False

logger = logging.getLogger("Tensoring Nets")
time_postfix = time.strftime("%a,%d_%b_%Y,%H:%M:%S", time.gmtime())
logs_name = 'logs/' + time_postfix + '.log'

fh = logging.FileHandler(logs_name) if SAVE_FILE_LOG else logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.setLevel(LOG_LEVEL)

device = torch.device("cuda")

import numpy as np


def matvec(M, N):
    logger.info(str(M.shape) + str(N.shape))
    assert (N.shape[1] == 1)
    res = np.zeros((N.shape[1], M.shape[0]))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            res[0][i] += M[i][j] * N[j][0]
    return res

