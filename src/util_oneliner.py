from math import log, exp
import numpy as np
import time, os, logging
from contextlib import contextmanager
def strlist_to_int(l):
    return [int(e) for e in l]

def log_sum_exp(seq):
    a=max(seq)
    return a+log(sum([exp(e-a) for e in seq]))

def element_wise_add(l1, l2):
    return map(lambda x,y: x+y, l1, l2)

def roughly_equal(f1, f2):
    return np.all(abs(f1-f2) < 1e-5)

def get_vocab_from_file(f):
    return dict((v, i) for i, v
         in enumerate(e.strip().split()[0] for e in f))

@contextmanager
def tictoc(name, logger=None):
    start_time=time.time()
    yield
    message="\n%s took </%0.3f> seconds.\n"%(
        name, time.time()-start_time)
    if logger is None:
        logging.debug(message)
    else:
        logger.debug(message)
    
def mean(it_e):
    total=0.0
    count=0
    for count,e in enumerate(it_e):
        total+=e
    return total/(count+1)

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
    return


def get_logger(name, filename):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger

def batcher(itr, batch_size):
    """ 
    >>> list(batcher(["I love\n","people who.", "Doh ! "], 2))
    [[['I', 'love'], ['people', 'who.']], [['Doh', '!']]]
    """
    l=[]
    for row in itr:
        e=row.strip().split()
        if len(e) > 0:
            l.append(e)
        if len(l)==batch_size:
            yield l
            l=[]
    yield l
