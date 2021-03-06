"""Test the score and gradient functions and the training procedures
"""
# Created: 25 October 2014
__author__  = "Pushpendre Rastogi"
__contact__ = "pushpendre@jhu.edu"
from nose import with_setup
from tag_order0hmm import *
def setup_func():
    "set up test fixtures"
    pass

def teardown_func():
    "tear down test fixtures"
    pass

# with_setup is useful only for test functions, not for test methods
# in unittest.TestCase subclasses or other test classes. For those
# cases, define setUp and tearDown methods in the class
@with_setup(setup_func, teardown_func)
def test():
    "test ..."
    pass

# Class-level setup fixtures may be named setup_class, setupClass,
# setUpClass, setupAll or setUpAll; teardown fixtures may be named
# teardown_class, teardownClass, tearDownClass, teardownAll or
# tearDownAll. Class-level setup and teardown fixtures must be class
# methods
def test_get_lp_from_natural_param():
    idx=2
    table=[1,2,3]
    from math import exp, log
    import numpy as np
    import nose
    assert_eq = nose.tools.assert_almost_equal
    gold_value = log(float(exp(table[idx]))/sum(exp(e) for e in table))
    calc_value = float(get_lp_from_natural_param(idx, np.array(table)).eval())
    assert_eq(gold_value, calc_value)

def test_batch_update_ao():
    pass
