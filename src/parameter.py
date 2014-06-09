import collections, itertools
from math import log
import numpy as np
from misc_util import log_sum_exp
class Add_Lambda_Smoother(collections.MutableMapping):
    """A dictionary that applies an arbitrary key-altering
    function before accessing the keys.
    This Smoother assumes that the first item is the outcome space
    and the rest are conditioned on. e.g. (a,b,c) means (a|b,c)
    It does not backoff to a smooth estimate if b,c is absent.
    It just returns 0.0 (because that's log of 1.0)"""

    def __init__(self, Lambda, V):
        """ Lambda is the constant to be added in the numerator
        V is the number of distinct possibilities of the discrete event
        We add lambda*V to the denominator
        """
        self.numerator=dict()
        self.denominator=dict()
        self.Lambda=Lambda
        self.V=V

    def __getitem__(self, key):
        if key in self.numerator:
            num=self.numerator[key]+self.Lambda
        else:
            num=self.Lambda
        k=key[1:]
        if k in self.denominator:
            den=self.denominator[k]+(self.Lambda * self.V)
        else:
            return 0.0
        return log(num)-log(den)

    def __setitem__(self, key, value):
        if key in self.numerator:
            self.numerator[key]+=value
        else:
            self.numerator[key]=value
        k=key[1:]
        if k in self.denominator:
            self.denominator[k]+=value
        else:
            self.denominator[k]=value

    def __delitem__(self, key):
        del self.numerator[key]
        del self.denominator[key[1:]]

    def __iter__(self):
        return iter(self.numerator) # HACK: Maybe add denominator too

    def __len__(self):
        return len(self.numerator)


class Parameter:
    BOS=0
    EOS=1
    NULLTAG=0
    
    def __init__(self,
                 word_vocab,
                 word_embedding,
                 EOS_embedding,
                 tag_vocab,
                 unsup_ll_factor,
                 regularization_factor,
                 sup_word,
                 sup_tag,
                 regularization_type,
                 bilinear_init_sigma=0.01,
                 t_given_w_lambda=0.1,
                 w_given_t_BOS_lambda=0.01):
        np.random.seed(42)
        self.T = len(tag_vocab) # Number of tags (not including NULLTAG)
        self.V = len(word_vocab) # Number of words (not including BOS, EOS)
        self.R = len(word_embedding[0]) # dimensionality of embeddings
        self.regularization_type=regularization_type
        self.unsup_ll_factor=unsup_ll_factor
        self.regularization_factor=regularization_factor
        self.tag_vocab=tag_vocab
        self.word_vocab=word_vocab
        self.EOS_embedding=EOS_embedding
        ###################################################################
        ## Initialize the embeddings array. It contains R rows, V+1 column
        ###################################################################
        self.RW = np.vstack((np.zeros((1, self.R)),
                             np.array(word_embedding),
                             EOS_embedding)).T
        assert self.RW.shape==(self.R, self.V+2)
        ###################################################################
        ## Initialize dict to map Word to its Index in the Embeddings array
        ###################################################################
        self.word_idx_dict={}
        self.word_idx_dict[self.BOS]=0
        for i, w in enumerate(word_vocab):
            self.word_idx_dict[w]=(i+1)
        self.word_idx_dict[self.EOS]=self.V+1
        ###################################################################
        ## Initialize dict to map Tag to its Index in the Embeddings array
        ###################################################################
        self.tag_idx_dict={}
        self.tag_idx_dict[self.NULLTAG]=0
        for i, t in enumerate(tag_vocab):
            self.tag_idx_dict[t]=i+1
        ###################################################################
        ## Initialize the recurrent log-linear model
        ###################################################################
        self.C1 = np.random.randn(self.R, self.R)*bilinear_init_sigma
        self.C2 = np.random.randn(self.R, self.R)*bilinear_init_sigma
        self.U1 = np.random.randn(self.R, self.T+1)*bilinear_init_sigma
        self.U2 = np.random.randn(self.R, self.T+1)*bilinear_init_sigma
        self.BW = np.random.randn(self.V+2, 1)*bilinear_init_sigma
        ###################################################################
        ## Initialize TAG|Word and Word | BOS CPDs 
        ###################################################################
        self.t_given_w = Add_Lambda_Smoother(t_given_w_lambda, self.T)
        self.w_given_t_BOS= Add_Lambda_Smoother(w_given_t_BOS_lambda, self.V)
        for word, tag in itertools.izip(sup_word, sup_tag):
            prev_tag=self.NULLTAG
            prev_word=self.BOS
            for i in xrange(len(tag)):
                self.t_given_w[tag[i], prev_tag, prev_word]=1
                prev_tag=tag[i]
                prev_word=word[i]
        for word, tag in itertools.izip(sup_word, sup_tag):
            self.w_given_t_BOS[word[0], tag[0], self.BOS]=1
        return

    def get_tag_idx(self, tag):
        return self.tag_idx_dict[tag]

    def get_word_idx(self, word):
        return self.word_idx_dict[word]
    
    def get_lp_t_given_w(self, tag, prev_tag, prev_word):
        """ prev_tag can be NULLTAG and prev_word can be BOS
        """
        return self.t_given_w[tag, prev_tag, prev_word]
    
    def get_lp_w_given_t_BOS(self, word, cur_tag):
        """ This is the probability of only the BOS
        """
        assert cur_tag != self.NULLTAG
        return self.w_given_t_BOS[word, cur_tag, self.BOS]

    def get_next_seq_embedding(self, prev_word_seq, prev_embedding):
        pw = prev_word_seq[-1]
        c1rwk = np.dot(self.C1, self.RW[:, self.get_word_idx(pw)])
        if prev_embedding is not None:
            c2rw02km1 = np.dot(self.C2, prev_embedding)
            next_seq_embedding = c1rwk + c2rw02km1
        else:
            next_seq_embedding = c1rwk
        return next_seq_embedding
    
    def get_lp_w_given_two_tag_and_word_seq( \
        self, word, cur_tag, prev_tag, prev_word_seq, seq_embedding):
        """ Note that the prev_tag can be NULLTAG. In that case
        prev_word_seq has only 1 word.
        """ 
        u1tk = self.U1[:, self.get_tag_idx(cur_tag)]
        u2tkm1 = self.U2[:, self.get_tag_idx(prev_tag)]
        r_pred = seq_embedding + u1tk + u2tkm1
        wi = self.get_word_idx(word)
        rw = self.RW[:, wi]
        bw = self.BW[wi]
        lognum = np.dot(rw, r_pred)+bw
        logden = log_sum_exp([np.dot(self.RW[:, i], r_pred)+self.BW[i]
                              for i in xrange(1,self.V+2)])
        return lognum-logden
        
        
    def serialize(self, filename):
        """ Save this object to file. """

    def regularization_contrib(self):
        """ Calculate the regularization based on the current parameters
        """
        if self.regularization_type=="l2":
            return 0.0 # TODO
        else:
            raise NotImplementedError

