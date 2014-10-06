from math import log, exp
import numpy as np
def strlist_to_int(l):
    return [int(e) for e in l]

def token_iterator(f):
    for l in f:
        yield l.strip().split()

def log_sum_exp(seq):
    a=max(seq)
    return a+log(sum([exp(e-a) for e in seq]))

def element_wise_add(l1, l2):
    return map(lambda x,y: x+y, l1, l2)

def roughly_equal(f1, f2):
    return np.all(abs(f1-f2) < 1e-5)

def get_vocab_from_file(f):
    dict((x[1], x[0])
         for x
         in enumerate(e.strip().split()[0]
                      for e
                      in f))
