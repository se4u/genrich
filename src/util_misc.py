def str2int(l):
    return [int(e) for e in l]

def word_vocab_and_embedding(word_vocab_file):
    word=[e.split(" ")[0] for e in word_vocab_file]
    word_vocab_file.seek(0)
    embedding=[str2int(e.strip().split(" ")[1:]) for e in word_vocab_file]
    return [word, embedding]

def get_input_iterator(f):
    """ Returns an iterator using yield
    f is a file handle"""
    for l in f:
        yield l.strip().split()
    

def log_sum_exp(seq):
    a=max(seq)
    return a+log(sum([exp(e-a) for e in seq]))
