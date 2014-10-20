"""Small tests to understand theano features
"""
# Created: 19 October 2014
__author__  = "Pushpendre Rastogi"
__contact__ = "pushpendre@jhu.edu"
import theano
from theano import shared
from theano import tensor as T
from theano import function
" Does it make sense to straight away make shared variables from python lists ?"
arr=np.ones((2,1), np.double)*0.5
lp_table=shared(arr, borrow=True)
lp_mat=shared(np.array([[1, 2], [3, 4]]))
print type(a.get_value())

"Does it make sense to define T.iscalar ?"
tag_idx=T.iscalar("tag_idx")
tag_lp=lp_table[tag_idx]
tf_get_tag_lp1=function([tag_idx], tag_lp)
print tf_get_tag_lp1(0)
tf_get_tag_lp2=function([tag_idx], lp_table[tag_idx])
print tf_get_tag_lp2(0)

"Does it make sense to get the row by just a single number ? Even if it does for sake of consistency I would use matlab syntax"

"Does it make sense to take dot between a row and a matrix ?"

