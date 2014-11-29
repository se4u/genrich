"""Functions that extend theano library
"""
__date__    = "23 November 2014"
__author__  = "Pushpendre Rastogi"
__contact__ = "pushpendre@jhu.edu"
import theano.tensor as T
def log_sum_exp(x):
    m=T.max(x, axis=0)
    return T.log(T.sum(T.exp(x - m))) + m

def log_softmax(idx, table):
    denominator=log_sum_exp(table)
    return table[idx] - denominator
