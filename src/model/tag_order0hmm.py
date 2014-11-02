"""A simple order 0 (mean field like) tagger with emission parameters
that are automatically tuned over both supervised and unsupervised
data.
"""
# Created: 13 October 2014
__author__  = "Pushpendre Rastogi"
__contact__ = "pushpendre@jhu.edu"
import math, theano, logging, sys, os #, itertools
from tag_baseclass import tag_baseclass
import numpy as np
import theano.tensor as T
from theano import shared, function
from theano.ifelse import ifelse
from warnings import warn
from yaml import load, dump
from yaml import CDumper as Dumper
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# emb means embedding
# App Hungarian
#   tg_ means theano graph object
#   tf_ means theano function object
def log_sum_exp(x):
    m=T.max(x, axis=0)
    return T.log(T.sum(T.exp(x - m))) + m

def log_softmax(idx, table):
    denominator=log_sum_exp(table)
    return table[idx] - denominator

def get_lp_from_natural_param(idx, table):
    # return T.sum(T.log(T.nnet.softmax(table))[0, idx])
    return T.sum(log_softmax(idx, table.flatten()))    

def order0_ll_score_given_word_and_tag(tg_tag_id, tg_word_id, tg_lp_tag_np_table,
                                       tg_tag_emb, tg_word_emb):
    tg_tag_word_dot=T.dot(tg_word_emb, tg_tag_emb[tg_tag_id, :])
    # Log probability of word given tag
    tg_lp_word_given_tag=get_lp_from_natural_param(tg_tag_id, tg_tag_word_dot)
    # Log probability of tag
    tg_lp_tag=get_lp_from_natural_param(tg_tag_id, tg_lp_tag_np_table)
    # Return the sum
    return tg_lp_tag + tg_lp_word_given_tag

def order0_ll_score_given_word_only(tg_word_id,
                                    tg_lp_tag_np_table, tg_tag_emb,
                                    tg_word_emb,num_tag):
    # Return the total 
    return log_sum_exp(theano.map(fn=order0_ll_score_given_word_and_tag,
                                  sequences=[T.arange(num_tag)],
                                  non_sequences=[tg_word_id,
                                                 tg_lp_tag_np_table,
                                                 tg_tag_emb,
                                                  tg_word_emb],
                                  name="order0_ll_score_map")[0])

def get_random_emb(vocab_size, embedding_size):
        emb_size=(vocab_size, embedding_size)
        emb_arr=np.random.random_sample(emb_size)
        return emb_arr

def get_cpd_type(cpd_type):
    if cpd_type.startswith("lbl"):
        return "lbl"
    elif cpd_type.startswith("addlambda"):
        return "addlambda"
    return

class tag_order0hmm(tag_baseclass):
    """ This model implements a very basic log-bilinear CPD and an order 0
    HMM. The main sections are the TRAIN, PREDICT and GENERATE. It also
    supports two types of objectives, LL and NCE.
    USAGE:
    # The first creates an object from the command line,
    # with a random embedding. The second one initializes
    # from a file along with embeddings.
    >>> self = tag_order0hmm("lbl10", "LL", "L1", 0.01, 0.5, False, None, dict(t0=0, t1=1, t2=2), dict(w0=0, w1=1, w2=2))
    >>> self = tag_order0hmm(None, None, None, None, None, True, r"res/order0hmm_param.yaml", None, None)
    """
    def get_penalty_for_lbl(self, test_time):
        if test_time==False:
            if self.param_reg_type=="L1": 
                penalty = -(self.param_reg_weight *
                            (T.sum(T.abs_(self._tg_tag_emb)) +
                             T.sum(T.abs_(self._tg_word_emb))))
            elif self.param_reg_type=="L2":
                penalty = -(self.param_reg_weight *
                            (T.sum(T.sqr(self._tg_tag_emb)) +
                             T.sum(T.sqr(self._tg_word_emb))))
            else:
                raise NotImplementedError(
                    "objective_type: %s, param_reg_type: %s, self.cpd_type: %s"%(
                        self.objective_type, self.param_reg_type, self.cpd_type))
        else:
            penalty = 0
        return penalty
    
    def __init__(self,
                 cpd_type,
                 objective_type,
                 param_reg_type,
                 param_reg_weight,
                 unsup_ll_weight,
                 init_from_file,
                 param_filename,
                 tag_vocab,
                 word_vocab,
                 test_time=True):
        """The cpd_type defines the type of cpd to use, including the
        smoothing and the model, hinton's lbl or structured sparse
        stuff. [lbl<int>, addlambda<float>]
        
        The objective_type is either [LL, NCE]
        The param_reg_type is either [L1, L2]
        param_reg_weight is weight of the regularization term
        unsup_ll_weight is a float that decides the weight given to
        the unsupervised training corpus compared to the supervised
        training examples. Its value is typically close to 1e-3

        tag_vocab is a dictionary that maps tags to indices
        word_vocab is a dictionary that maps words to indices.
        """
        if init_from_file:
            data=load(open(param_filename, "rb"))
            self.tag_vocab=data["tag_vocab"]
            self.word_vocab=data["word_vocab"]
            self.cpd_type=data["cpd_type"]
            self.num_tag=len(self.tag_vocab)
            self.num_word=len(self.word_vocab)
            self._embedding_size=data["_embedding_size"]
            self.tag_emb_arr=np.loads(data["tag_emb_arr"])
            self.word_emb_arr=np.loads(data["word_emb_arr"])
            self._lambda=data["_lambda"]
            self.objective_type=data["objective_type"]
            self.param_reg_type=data["param_reg_type"]
            self.unsup_ll_weight=data["unsup_ll_weight"]
            self.param_reg_weight=data["param_reg_weight"]
            self.tag_np_arr=data["tag_np_arr"]
        else:
            self.tag_vocab=tag_vocab
            self.word_vocab=word_vocab
            self.cpd_type=get_cpd_type(cpd_type)
            self.num_tag=len(tag_vocab)
            self.num_word=len(word_vocab)
            self._embedding_size=None
            self.tag_emb_arr=None
            self.word_emb_arr=None
            self._lambda=None
            if self.cpd_type=="lbl":
                self._embedding_size=int(cpd_type[3:])
                self.tag_emb_arr=get_random_emb(self.num_tag,
                                                self._embedding_size)
                self.word_emb_arr=get_random_emb(self.num_word,
                                                 self._embedding_size)
            elif self.cpd_type=="addlambda":
                self._lambda=float(cpd_type[9:])
            self.objective_type=objective_type
            self.param_reg_type=param_reg_type
            self.unsup_ll_weight=unsup_ll_weight 
            self.param_reg_weight= param_reg_weight
            lp_tag_np=math.log(1/float(self.num_tag))
            self.tag_np_arr=np.ones((1, self.num_tag),np.float)*lp_tag_np
        
        self._tg_lp_tag_np_table=shared(self.tag_np_arr, borrow=True)
        warn("update tg_lp_tag_np_table using its set_value method only")
        if self.cpd_type=="lbl":
            self._tg_tag_emb=shared(self.tag_emb_arr, borrow=True)
            self._tg_word_emb=shared(self.word_emb_arr, borrow=True)
            self.params=[self._tg_lp_tag_np_table,
                         self._tg_tag_emb,
                         self._tg_word_emb]
            warn("Update the _tg_tag_emb, _tg_word_emb using only set_value")
        elif self.cpd_type=="addlambda":
            self.params=None
            raise NotImplementedError
        
        # Now I'd use the objective and unsup_ll_weight
        self.unsup_ll_weight=T.constant(self.unsup_ll_weight,
                                        name="unsup_ll_weight",
                                        dtype=np.float)
        self.param_reg_weight=T.constant(self.param_reg_weight,
                                         name="param_reg_weight",
                                         dtype=np.float)
        tag_ids=T.ivector("tag_ids")
        word_ids=T.ivector("word_ids")
        if self.objective_type=="LL" and self.cpd_type=="lbl":
            # self.tg_score_sto=self.score_sto_ll_tg(tag_ids, word_ids)+\
            #     self.get_penalty_for_lbl(test_time)
            tg_score_ao=self._score_ao_tg(tag_ids, word_ids)+\
                self.get_penalty_for_lbl(test_time)
            if test_time:
                tg_score_so=self._score_so_tg(tag_ids, word_ids)
            else:
                tg_score_so=self._score_so_tg(word_ids)*\
                    self.unsup_ll_weight + self.get_penalty_for_lbl(test_time)
            
            self.tg_gradient_ao=T.grad(tg_score_ao,self.params)
            self.tg_gradient_so=T.grad(tg_score_so,self.params)
            
            self._score_ao=function([tag_ids, word_ids], tg_score_ao,
                                    name="_score_ao")
            self._score_so=function([word_ids], tg_score_so,
                                    name="_score_so")
            eta=T.fscalar("eta")
            
            self._update_ao=function([eta, tag_ids, word_ids],
                                     self.tg_gradient_ao,
                                     name="_update_ao",
                                     updates=[(p, p+eta*g)
                                              for (g, p)
                                              in zip(self.tg_gradient_ao,
                                                     self.params)],
                                     )
                                     
            self._update_so=function([eta, word_ids], self.tg_gradient_so,
                                     name="_update_so",
                                     updates=[(p, p+eta*g)
                                              for (g, p)
                                              in zip(self.tg_gradient_so,
                                                     self.params)]
                                     )
            
        else:
             raise NotImplementedError(
                 "objective_type: %s, self.cpd_type: %s"%(
                     objective_type, self.cpd_type))
        return

    def _score_ao_tg(self, tag_ids, word_ids):
        output, _ = theano.map(fn=order0_ll_score_given_word_and_tag, 
                               sequences=[tag_ids, word_ids], 
                               non_sequences=[self._tg_lp_tag_np_table,
                                              self._tg_tag_emb,
                                              self._tg_word_emb],
                               name="_score_ao_tg")
        return T.sum(output)
    
    def _score_so_tg(self, word_ids):
        """sto means sentence and tag observed.
        This function returns a tg object. which can then be compiled
        (or further operated on) . This function assumes that its words
        and tags are integers
        """
        output, _ = theano.map(fn=order0_ll_score_given_word_only, 
                               sequences=[word_ids], 
                               non_sequences=[self._tg_lp_tag_np_table,
                                              self._tg_tag_emb,
                                              self._tg_word_emb,
                                              self.num_tag],
                               name="_score_so_tg")
        return T.sum(output)
    
    def score_ao(self, tags, words):
        """This function receives actual words and tags (not indices!) 
        and then returns their probabilties.
        """
        return self._score_ao([self.get_from_tag_vocab(t) for t in tags],
                              [self.get_from_word_vocab(w) for w in words])
    
    def score_so(self, words):
        return self._score_so([self.get_from_word_vocab(w) for w in words])
    
    def update_ao(self, eta, tags, words):
        """ The update when both tags and sentences are observed.
        Note that in the case of cpd_type="lbl" we return a list with
        gradients for self.params
        """
        
        return self._update_ao(eta,
                               [self.get_from_tag_vocab(t) for t in tags],
                               [self.get_from_word_vocab(w) for w in words])
    
    def update_so(self, eta, words):
        """ This is the update for unsupervised examples.
        """
        return self._update_so(eta,
                               [self.get_from_word_vocab(w) for w in words])
    
    def get_perplexity(self, sentence):
        "Return the actual perplexity of the unsupervised data"
        if self.objective_type == "LL":
            return self.score_so(sentence)/self.unsup_ll_weight
        else:
            raise NotImplementedError
    
    # PREDICT
    def predict_posterior_tag_sequence(self, words):
        """ Given the input words which tag would be most appropriate
        for the word at a particular position. This function finds
        that tag sequence. 
        """
        tags=[None]*len(words)
        score=[None]*len(words)
        for i,word in enumerate(words):
            wi=self.get_from_word_vocab(word)
            tags[i], score[i]=max(((tag, self._score_ao([ti], [wi]))
                         for (tag, ti) in self.tag_vocab.iteritems()
                         ),
                        key=lambda x: x[1])
            score[i]=float(score[i])
        return tags, score
    
    def predict_viterbi_tag_sequence(self, words):
        """ Given the input words which sequence of tags would
        maximize the perplexity. This function finds that sequence.
        In the order0 case it would simply be the posterior_tag_sequence
        """
        return self.predict_posterior_tag_sequence(words)

    # GENERATE
    def generate_word_tag_sequence(self):
        """ Generate a sequence of tags, and then (word|tags).
        Assume that "." is the EOS tag. As soon as you generate "."
        stop generating further
        """
        # NOTE: self._tg_lp_tag_np_table_arr contains natural parameters for the log probabilities
        # There should be a trick to sample from that !
        raise NotImplementedError
        while 1:
            # Draw a tag.
            # Use the
            if self.cpd_type=="lbl":
                pass
            elif self.cpd_type=="addlambda":
                pass
            break
        return
    
    # Following functions are utility functions
    def save(self, filename):
        """Save the object's attributes to the yaml filename
        """
        
        f=open(filename, "wb")
        try:
            f.write(dump(dict(
                        tag_vocab=self.tag_vocab,
                        word_vocab=self.word_vocab,
                        cpd_type=self.cpd_type,
                        _embedding_size=self._embedding_size,
                        tag_np_arr=self.tag_np_arr.dumps(),
                        tag_emb_arr=self.tag_emb_arr.dumps(),
                        word_emb_arr=self.word_emb_arr.dumps(),
                        _lambda=self._lambda,
                        objective_type=self.objective_type,
                        param_reg_type=self.param_reg_type,
                        unsup_ll_weight= \
                            self.unsup_ll_weight.get_scalar_constant_value(),
                        param_reg_weight= \
                            self.param_reg_weight.get_scalar_constant_value()
                        ),
                         Dumper=Dumper,
                         default_flow_style=False))
        except Exception as exc:
            f.close()
            os.remove(filename)
            print exc
        finally:
            f.close()
                
        
    def get_from_word_vocab(self, word):
        try:
            return self.word_vocab[word]
        except:
            return self.word_vocab["<OOV>"]
        
    def get_from_tag_vocab(self, tag):
        try:
            return self.tag_vocab[tag]
        except:
            return self.tag_vocab["<OOV>"]
        
    
