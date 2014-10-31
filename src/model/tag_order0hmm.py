"""A simple order 0 (mean field like) tagger with emission parameters
that are automatically tuned over both supervised and unsupervised
data.
"""
# Created: 13 October 2014
__author__  = "Pushpendre Rastogi"
__contact__ = "pushpendre@jhu.edu"
import math, yaml, theano, logging
from tag_baseclass import tag_baseclass
import numpy as np
import theano.tensor as T
from theano import shared, function
from theano.ifelse import ifelse
from warnings import warn

debug_logger=logging.getlogger("debug")
debug_logger.addHandler(logging.StreamHandler)
debug_logger.setLevel(logging.DEBUG)

# emb means embedding
# App Hungarian
#   tg_ means theano graph object
#   tf_ means theano function object
def order0_ll_score_given_word_and_tag(tg_tag_id, tg_word_id, tg_lp_tag_table,
                                       tg_tag_emb, tg_word_emb):
    tg_tag_word_dot=T.dot(tg_word_emb, tg_tag_emb[tg_tag_id, :])
    tg_lp_word_given_tag_table=T.log(T.nnet.softmax(tg_tag_word_dot))
    # The extra 0 is because the softmax creates a row tensor
    tg_lp_word_given_tag=tg_lp_word_given_tag_table[0, tg_word_id]
    tg_lp_tag=tg_lp_tag_table[tg_tag_id]
    return T.sum(tg_lp_tag + tg_lp_word_given_tag)

def log_sum_exp(x):
    debug_logger.debug("ran log_sum_exp")
    return T.log(T.sum(T.exp(x - T.max(x)))) + T.max(x)

def order0_ll_score(tg_tag_id, tg_word_id, tg_lp_tag_table, tg_tag_emb, tg_word_emb,
                    num_tag):
    warn(" ifelse had better be lazy evaluated !!")
    return ifelse(T.gt(tg_word_id, -1),
                  order0_ll_score_given_word_and_tag(tg_tag_id, tg_word_id,
                                                     tg_lp_tag_table,
                                                     tg_tag_emb, tg_word_emb),
                  log_sum_exp(theano.map(fn=order0_ll_score_given_word_and_tag,
                                   sequences=[T.arange(num_tag)],
                                   non_sequences=[tg_word_id,
                                                  tg_lp_tag_table,
                                                  tg_tag_emb,
                                                  tg_word_emb],
                                   name="order0_ll_score_map")[0])
                  )

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
                 training_method):
        """The cpd_type defines the type of cpd to use, including the
        smoothing and the model, hinton's lbl or structured sparse
        stuff. [lbl<int>, addlambda<float>]
        
        The objective_type is either [LL, NCE]
        The param_reg_type is either [L1, L2]

        unsup_ll_weight is a float that decides the weight given to
        the unsupervised training corpus compared to the supervised
        training examples. Its value is typically close to 1e-3

        tag_vocab is a dictionary that maps tags to indices
        word_vocab is a dictionary that maps words to indices

        training_method is decides the update equation. It is either
        [add_projected, add_naturalparam, mult_exponentiated, mult_prod]
        add_projected      | additive sgd with probabilities and
                           | then reprojects them back to probability land
        add_naturalparam   | additive sgd on a multinomial expressed as
                           | the natural parameters
        mult_exponentiated | multiplicative exponentiated sgd with
                           | probabilities
        mult_prod          | multiplicative prod SGD. 
        """
        if init_from_file:
            data=yaml.load(open(param_filename, "rb"))
            self.tag_vocab=data["tag_vocab"]
            self.word_vocab=data["word_vocab"]
            self.cpd_type=data["cpd_type"]
            self.num_tag=len(tag_vocab)
            self.num_word=len(word_vocab)
            self._embedding_size=data["_embedding_size"]
            tag_emb_arr=np.array(data["tag_emb_arr"], dtype=np.double)
            word_emb_arr=np.array(data["word_emb_arr"], dtype=np.double)
            self._lambda=data["_lambda"]
            self.objective_type=data["objective_type"]
            self.param_reg_type=data["param_reg_type"]
            unsup_ll_weight=data["unsup_ll_weight"]
            param_reg_weight=data["param_reg_weight"]
        else:
            self.tag_vocab=tag_vocab
            self.word_vocab=word_vocab
            self.cpd_type=get_cpd_type(cpd_type)
            self.num_tag=len(tag_vocab)
            self.num_word=len(word_vocab)
            if self.cpd_type=="lbl":
                self._embedding_size=int(cpd_type[3:])
                tag_emb_arr=get_random_emb(self.num_tag, self._embedding_size)
                word_emb_arr=get_random_emb(self.num_word, self._embedding_size)
            elif self.cpd_type=="addlambda":
                self._lambda=float(cpd_type[9:])
            self.objective_type=objective_type
            self.param_reg_type=param_reg_type
            # unsup_ll_weight (Just for visual correspondence)
            # param_reg_weight
        
        lp_tag=math.log(1/float(self.num_tag))
        self._tg_lp_tag_table=shared(np.ones((self.num_tag, 1),np.double)*lp_tag)
        warn("update tg_lp_tag_table using its set_value method only")
        if self.cpd_type=="lbl":
            self._tg_tag_emb=shared(tag_emb_arr, borrow=True)
            self._tg_word_emb=shared(word_emb_arr, borrow=True)
            warn("Update the _tg_tag_emb, _tg_word_emb using only set_value")
        elif self.cpd_type=="addlambda":
            raise NotImplementedError
        
        # Now I'd use the objective and unsup_ll_weight
        self.unsup_ll_weight=T.constant(unsup_ll_weight, name="unsup_ll_weight",
                                        dtype=np.double)
        self.param_reg_weight=T.constant(param_reg_weight, name="param_reg_weight",
                                         dtype=np.double)
        tag_ids=T.ivector("tag_ids")
        word_ids=T.ivector("word_ids")
        if objective_type=="LL" and self.cpd_type=="lbl":
            if param_reg_type=="L1": 
                self.tg_score_sto= self.score_sto_ll_tg(tag_ids, word_ids) - \
                    (self.param_reg_weight *
                     (T.sum(T.abs_(self._tg_tag_emb)) +
                      T.sum(T.abs_(self._tg_word_emb))
                      )
                     )
            elif param_reg_type=="L2":
                self.tg_score_sto= self.score_sto_ll_tg(tag_ids, word_ids) - \
                    (self.param_reg_weight *
                     (T.sum(T.sqr(self._tg_tag_emb)) +
                      T.sum(T.sqr(self._tg_word_emb))
                      )
                     )
            else:
                raise NotImplementedError(
                    "objective_type: %s, param_reg_type: %s, self.cpd_type: %s"%(
                        objective_type, param_reg_type, self.cpd_type))
            tg_score_ao=self.tg_score_sto
            tg_score_so=self.tg_score_sto*self.unsup_ll_weight
            tg_gradient_ao=T.grad(tg_score_ao,[self._tg_lp_tag_table,
                                               self._tg_tag_emb,
                                               self._tg_word_emb])
            tg_gradient_so=T.grad(tg_score_so, [self._tg_lp_tag_table,
                                                self._tg_tag_emb,
                                                self._tg_word_emb])
            # Basically so has the same signature you just put -1 for tag-ids
            self._score_basic=function([tag_ids, word_ids],
                                      self.score_sto_ll_tg(tag_ids, word_ids))
            self._score_ao=function([tag_ids, word_ids], tg_score_ao)
            self._score_so=function([tag_ids, word_ids], tg_score_so)
            self._gradient_ao=function([tag_ids, word_ids], tg_gradient_ao)
            self._gradient_so=function([tag_ids, word_ids], tg_gradient_so)
        else:
             raise NotImplementedError(
                 "objective_type: %s, self.cpd_type: %s"%(
                     objective_type, self.cpd_type))
        return
    
    def score_sto_ll_tg(self, tag_ids, word_ids):
        """sto means sentence and tag observed.
        This function returns a tg object. which can then be compiled
        (or further operated on) 
        """
        output, _ = theano.map(fn=order0_ll_score, 
                               sequences=[tag_ids, word_ids], 
                               non_sequences=[self._tg_lp_tag_table,
                                              self._tg_tag_emb,
                                              self._tg_word_emb,
                                              self.num_tag],
                               name="score_sto_ll_tg_map")
        return T.sum(output)
    
    def score_ao(self, tags, words):
        return self._score_ao([self.tag_vocab[t] for t in tags],
                              [self.word_vocab[w] for w in words])
    
    def score_so(self, words):
        return self._score_ao([-1]*len(words),
                              [self.word_vocab[w] for w in words])
    
    def gradient_ao(self, tags, words):
        """ The gradient when both tags and sentences are observed.
        Note that in the case of cpd_type="lbl" we return a list with
        gradients for the following parameters
        [self._tg_lp_tag_table, self._tg_tag_emb, self._tg_word_emb]
        """
        return self._gradient_ao([self.tag_vocab[t] for t in tags],
                                 [self.word_vocab[w] for w in words])
    
    def gradient_so(self, words):
        """
        """
        return self._gradient_so([-1]*len(words),
                                 [self.word_vocab[w] for w in words])
    
    def get_perplexity(self, sentence):
        "Return the actual perplexity of the unsupervised data"
        if self.objective_type == "LL":
            return self.score_so(sentence)/self.unsup_ll_weight
        else:
            raise NotImplementedError
    
    def set_parameter(self, pd):
        """ pd means parameter dictionary
        """
        if self.cpd_type=="lbl":
            self._tg_lp_tag_table.set_value(pd["lp_tag_table"])
            self._tg_tag_emb.set_value(pd["tag_emb"])
            self._tg_word_emb.set_value(pd["word_emb"])
        else:
            raise NotImplementedError
        return
    
    def get_parameter(self, borrow=False):
        """This function returns me a COPY of the current params.
        It includes EVERYTHING! The params used in the lm,
        the params used for state transitions of the tagging model.

        Do not rely on borrowing happening all the time !!
        """
        if self.cpd_type=="lbl":
            return dict(lp_tag_table=self._tg_lp_tag_table.get_value(borrow=borrow),
                        tag_emb=self._tg_tag_emb.get_value(borrow=borrow),
                        word_emb=self._tg_word_emb.get_value(borrow=borrow)
                        )
        else:
            raise NotImplementedError
    
    # TRAIN
    def stochastic_train(self, eta, tags=None, words=None):
        """ Given the learning rate eta, words and corresponding tags
        update the parameters using the SGD update. Note that eta is a
        positive float with a small value.
        
        Also note that tags can be None but words cannot be.
        """
        # THIS FUNCTION HAS NOT BEEN TESTED. TEST THAT THE UPDATES
        # ARE CORRECT. Also renormalize the log-probabilities so they add upto 0
        # THIS GOES INTO EXPONENTIATED GRADIENT, PROJECTED GRADIENT ETC.
        # DO THIS IN A WAY THAT YOU DONT FALL INTO LOCAL OPTIMA
        # projected gradient is fastest to move currently small
        # probabilities, when the objective calls for it, while GD in
        # logspace is slowest to move them.  EG is in between. Because
        # Projected gradient does φ += ε ∂F/∂φ (followed by additive
        # renormalization).  EG scales the update size by a factor of
        # φ, since for small ε, the EG update φ *= exp ε ∂F/∂φ is
        # close to φ += ε φ ∂F/∂φ (followed by multiplicative
        # renormalization).  GD in logspace adds another factor of φ
        # (after shifting by E).
        # EG can actually be viewed as a projected subgradient method
        # using generalized relative entropy (D(x || y) = \sum_i x_i
        # log (x_i/y_i) - x_i + y_i ) as the distance function for
        # projections (Beck & Teboulle, 2003)
        assert words is not None
        if self.cpd_type=="lbl":
            if tags is None:
                grad_lp_tag_table, grad_tag_emb, grad_word_emb = \
                    self.gradient_so(words)
            else:
                grad_lp_tag_table, grad_tag_emb, grad_word_emb = \
                    self.gradient_ao(tags, words)
            
            lp_tags = self._tg_lp_tag_table.get_value()+grad_lp_tag_table
            lp_tags = lp_tags - lp_tags.mean(axis=0)
            print self._tg_lp_tag_table.get_value(), grad_lp_tag_table
            self._tg_lp_tag_table.set_value(lp_tags)
            print self._tg_lp_tag_table.get_value()
            self._tg_tag_emb.set_value(self._tg_tag_emb.get_value()+grad_tag_emb)
            self._tg_word_emb.set_value(self._tg_word_emb.get_value()+grad_word_emb)
        else:
            raise NotImplementedError

    # PREDICT
    def predict_posterior_tag_sequence(self, words):
        """ Given the input words which tag would be most appropriate
        for the word at a particular position. This function finds
        that tag sequence. 
        """
        tags=[None]*len(words)
        for i,word in enumerate(words):
            wi=self.word_vocab[word]
            tags[i]=max(((tag, self.score_ao([ti], [wi]))
                         for (tag, ti) in self.tag_vocab.iteritems()
                         ),
                        key=lambda x: x[1])
        return tags

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
        # NOTE: self._tg_lp_tag_table_arr contains log probabilities
        # of a multinomial over tags. There should be a trick to
        # sample from that !
        raise NotImplementedError
        while 1:
            # Draw a tag.
            # Use the
            if self.cpd_type=="lbl":
                pass
            elif self.cpd_type=="addlambda":
                pass
            break
    
