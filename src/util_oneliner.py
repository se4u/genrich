from math import log, exp
import numpy as np
import time, sys
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
def tictoc(name, stream=sys.stderr):
    start_time=time.time()
    yield
    stream.write("\n%s took </%0.3f> seconds.\n"%(
            name, time.time()-start_time))

def mean(it_e):
    total=0.0
    count=0
    for count,e in enumerate(it_e):
        total+=e
    return total/(count+1)
    
