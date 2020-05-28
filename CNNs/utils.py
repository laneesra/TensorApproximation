import time
import logging
import torch

from itertools import repeat
from torch._six import container_abcs


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_pair = _ntuple(2)


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

device_name = 'cpu'
device = torch.device(device_name)

