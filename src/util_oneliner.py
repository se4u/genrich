from math import log, exp
import numpy as np
import time, os, logging, sys
from contextlib import contextmanager
import theano.tensor as T
from util_theano_extension import log_softmax

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
        logger=logging
    print >> sys.stderr, message
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

def get_lp_from_natural_param(idx, table):
    """ STATUS: Tested
    """
    return T.sum(log_softmax(idx, table.flatten()))    

def get_random_emb(vocab_size, embedding_size):
    """ The fact that is has a variance equal to inverse of fan-in and
    that it is a mean zero guassian at the beginning is very important 
    """
    emb_arr=np.random.randn(vocab_size,
                            embedding_size)/pow(vocab_size, 0.5)
    return emb_arr
global __mycache
__mycache={};
def cache(tag, f, param):
    global __mycache
    try:
        return __mycache[(tag,param)]
    except:
        v=f(*(param))
        __mycache[(tag,param)]=v
        return v

global __mypcache
__mypcache={}
def purgable_cache(tag, f, param):
    global __mypcache
    try:
        return __mypcache[(tag,param)]
    except:
        v=f(*(param))
        __mypcache[(tag,param)]=v
        return v

def purge_cache():
    global __mypcache
    del __mypcache
    __mypcache={}
