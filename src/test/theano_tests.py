"""Small tests to understand theano features
"""
# Created: 19 October 2014
__author__  = "Pushpendre Rastogi"
__contact__ = "pushpendre@jhu.edu"
from theano.ifelse import ifelse
from theano import shared, function, Mode
from theano import tensor as T
import timeit
import numpy as np
from math import log
" Does it make sense to straight away make shared variables from python lists ?"
arr=np.ones((2,1), np.double)*0.5
lp_table=shared(arr, borrow=True)
lp_mat=shared(np.array([[1, 2], [3, 4]]))
print type(arr.get_value())

"Does it make sense to define T.iscalar ?"
tag_idx=T.iscalar("tag_idx")
tag_lp=lp_table[tag_idx]
tf_get_tag_lp1=function([tag_idx], tag_lp)
print tf_get_tag_lp1(0)
tf_get_tag_lp2=function([tag_idx], lp_table[tag_idx])
print tf_get_tag_lp2(0)

"is ifelse lazy evaluated ?"
a = T.scalar('a')

def g(a):
    return T.sum(T.arange(a)) # compare with sum(range(a))

def f(a):
    return T.sum(T.arange(a)) # compare with sum(range(a))

f1=function([a], T.switch(T.gt(1,1), f(a), g(a+1)), mode=Mode(linker='cvm'), on_unused_input='ignore')
g1=function([a], ifelse(T.gt(1,1), f(a), g(a+1)), mode=Mode(linker='cvm'), on_unused_input='ignore')
timeit.timeit('f1(100000)', "from __main__ import f1", number=10000)
timeit.timeit('g1(100000)', "from __main__ import g1", number=10000)
# You must ensure that only theano ops are in the graph. Nothing else that might force an actual compilation before hand and destroy the laziness. 
"Problem with gradient computation in Theano with ifelse"
import theano
from theano import function, tensor
from theano.ifelse import ifelse


i = tensor.iscalar('i')
# Vector of i elements, each equal to i
a = tensor.alloc(i.astype('float64'), i)
m = a.max()

out = ifelse(i <= 0,
             T.constant(-1, dtype='float64'),
             m)

f_out = function([i], out, mode=Mode(linker='cvm'))
theano.printing.debugprint(f_out, print_type=True)
print f_out(3)
print f_out(0)
print f_out(-1)

dout_di = theano.grad(out, i)
f_grad = theano.function([i], dout_di, mode=Mode(linker='cvm'))
theano.printing.debugprint(f_grad, print_type=True)
print f_grad(3)
print f_grad(0)
print f_grad(-1)
# One way it might be possible (but maybe not a good idea) would be to
# add a special case in theano.grad, so that if an intermediate gradient
# has the form og=ifelse(c, DisconnectedType, ig), we backpropagate using
# ifelse(c, DisconnectedType, op.grad(ig)) instead of op.grad(og). 
"Does it make sense to get the row by just a single number ? Even if it does for sake of consistency I would use matlab syntax"

"Does it make sense to take dot between a row and a matrix ?"


