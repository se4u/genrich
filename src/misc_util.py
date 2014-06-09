from math import log, exp
import numpy as np
def str2int(l):
    return [int(e) for e in l]

def word_vocab_and_embedding(word_vocab_file):
    word=[e.split(" ")[0] for e in word_vocab_file]
    assert word[-1]==r"</s>"
    word=word[:-1]
    word_vocab_file.seek(0)
    embedding=[str2int(e.strip().split(" ")[1:]) for e in word_vocab_file]
    return [word, embedding[:-1], embedding[-1]]

def get_input_iterator(f):
    """ Returns an iterator using yield
    f is a file handle"""
    for l in f:
        yield l.strip().split()
    

def log_sum_exp(seq):
    a=max(seq)
    return a+log(sum([exp(e-a) for e in seq]))


def list_add(l1, l2):
    return map(lambda x,y: x+y, l1, l2)

def close(f1, f2):
    return np.all(abs(f1-f2) < 1e-5)
