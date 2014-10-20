"""A simple order 0 (mean field like) tagger with emission parameters
that are automatically tuned over both supervised and unsupervised
data.
"""
# Created: 13 October 2014
__author__  = "Pushpendre Rastogi"
__contact__ = "pushpendre@jhu.edu"
import math
from tag_baseclass import tag_baseclass
import numpy as np
import theano.tensor as T
from theano import shared, function
import theano
from warnings import warn
# emb means embedding
# App Hungarian
#   tg_ means theano graph object
#   tf_ means theano function object

def order0_ll_score_given_word_and_tag(tg_tag_id, tg_word_id, tg_lp_tag_table,
                                       tg_tag_emb, tg_word_emb):
    tg_tag_word_dot=T.dot(tg_word_emb, tg_tag_emb[tg_tag_id, :])
    tg_lp_word_given_tag_table=T.log(T.nnet.softmax(tg_tag_word_dot))
    tg_lp_word_given_tag=tg_lp_word_given_tag_table[tg_word_id]
    tg_lp_tag=tg_lp_tag_table[tg_tag_id]
    return tg_lp_tag + tg_lp_word_given_tag

def order0_ll_score(tg_tag_id, tg_word_id, tg_lp_tag_table, tg_tag_emb, tg_word_emb,
                    num_tag):
    return theano.ifelse.ifelse(
        theano.map(fn=order0_ll_score,
                   sequences=[T.arange(num_tag)],
                   non_sequences=[tg_word_id, tg_lp_tag_table, tg_tag_emb,
                                  tg_word_emb, num_tag],
                   name="order0_ll_score_map"),
        order0_ll_score_given_word_and_tag(tg_tag_id, tg_word_id, tg_lp_tag_table,
                                           tg_tag_emb, tg_word_emb)
        )
    
class tag_order0hmm(tag_baseclass):
    def __init__(self,
                 cpd_type,
                 objective_type,
                 param_reg_type,
                 param_reg_weight,
                 unsup_ll_weight,
                 initialize_param_according_to_file,
                 word_embedding_filename,
                 tag_vocab,
                 word_vocab):
        """The cpd_type defines the type of cpd to use, including the
        smoothing and the model, hinton's lbl or structured sparse
        stuff. [lbl<int>, unsmooth, addlambda<float>]
        
        The objective_type is either [LL, NCE]

        The param_reg_type is either [L1, L2]

        unsup_ll_weight is a float that decides the weight given to
        the unsupervised training corpus compared to the supervised
        training examples. Its value is typically close to 1e-3

        tag_vocab is a dictionary that maps tags to indices

        word_vocab is a dictionary that maps words to indices
        """
        self.tag_vocab=tag_vocab
        self.word_vocab=word_vocab
        self.num_tag=len(tag_vocab)
        self.num_word=len(word_vocab)
        lp_tag=math.log(1/self.num_tag)
        self._tg_lp_tag_table=shared(np.ones((self.num_tag, 1),np.double)*lp_tag)
        warn("update tg_lp_tag_table using its set_value method only")
        # tg_tag_idx=T.iscalar("tag_idx")
        # self.tg_lp_tag=self.tg_lp_tag_table[tg_tag_idx]
        # self.tf_get_lp_tag=function([tg_tag_idx], self.tg_lp_tag)
        if cpd_type.startswith("lbl"):
            self.cpd_type="lbl"
            self._embedding_size=int(cpd_type[2:])
            self._tg_tag_emb=shared(self._get_random_emb(self.num_tag))
            self._tg_word_emb=shared(self._get_random_emb(self.num_word))
            warn("Update the _tg_tag_emb, _tg_word_emb using only set_value")
            # Make new tag_idx objects to aid in debugging later
            # _tf_get_lp_word_given_tag takes a tag idx and returns lp
            # tg_tag_idx2=T.iscalar("tag_idx2")
            # tg_word_idx=T.iscalar("word_idx")
            # tg_tag_word_dot=T.dot(self._tg_word_emb,
            #                       self._tg_tag_emb[tg_tag_idx2, :])
            # tg_lp_word_given_tag_table=T.log(T.nnet.softmax(tg_tag_word_dot))
            # self.tg_lp_word_given_tag=tg_lp_word_given_tag_table[tg_word_idx]
            # self.tf_get_lp_word_given_tag=function([tg_word_idx, tg_tag_idx],
            #                                        self.tg_lp_word_given_tag)
        elif cpd_type == "unsmooth":
            self.cpd_type="unsmooth"
            word_given_lp_tag=math.log(1/self.num_word)
            word_given_lp_tag_arr=np.ones((self.num_tag, self.num_word),
                                          np.double)*word_given_lp_tag
            self._tg_lp_word_given_tag_table=shared(word_given_lp_tag_arr)
            word_idx2=T.iscalar("word_idx2")
            tag_idx3=T.iscalar("tag_idx3")
            self.tf_get_lp_word_given_tag=function(
                [word_idx2, tag_idx3],
                self._tf_get_lp_word_given_tag_table[tag_idx3, word_idx2])
        elif cpd_type.startswith("addlambda"):
            addlambda=float(cpd_type[9:])
            self.addlambda=T.constant(addlambda, name="addlambda",
                                       dtype=np.double)
            raise NotImplementedError

        "Now I'd use the objective and unsup_ll_weight"
        self.objective_type=objective_type
        self.unsup_ll_weight=unsup_ll_weight
        self.param_reg_type=param_reg_type
        self.param_reg_weight=param_reg_weight
        if objective_type=="LL" and param_reg_type=="L1" and self.cpd_type=="lbl":
            self.score_sto_tg=self.score_sto_ll_tg
        elif objective_type=="LL" and param_reg_type=="L2" and self.cpd_type=="lbl":
            raise NotImplementedError("TODO")
        else:
            raise NotImplementedError(
                "objective_type: %s, param_reg_type: %s, self.cpd_type: %s"%(
                    objective_type, param_reg_type, self.cpd_type))
        if initialize_param_according_to_file:
            raise NotImplementedError
    
    def _get_random_emb(self, vocab_size):
        emb_size=(vocab_size, self._embedding_size)
        emb_arr=np.random.random_sample(emb_size)
        return emb_arr

    
    def score_sto_ll_tg(self):
        """sto means sentence and tag observed.
        This function returns a tg object. which can then be compiled
        (or further operated on) 
        """
        tag_ids=T.ivector("tag_id")
        word_ids=T.ivector("word_id")
        return theano.map(fn=order0_ll_score, 
                          sequences=[tag_ids, word_ids], 
                          non_sequences=[self._tg_lp_tag_table,
                                         self._tg_tag_emb,
                                         self._tg_word_emb,
                                         self.num_tag],
                          name="score_sto_ll_tg_map")
        
    
    def score_ao(self, sentence, tag):
        """ao means all observed. That both the tag and the word were observed
        """
        

    def score_so(self, sentence):
        """so means that the sentence was observed, but not the tags
        This requires the use of DP. We are calculating the
        E[P(sentence)] given the parameters.
        This only needs a forward pass to compute the total probability.
        """
        sentence_id=[self.word_vocab[e] for e in sentence]
        tag_id=[-1]*len(sentence)
        
        tf=function()
        return tf()

    def gradient_so(self, sentence):
        raise NotImplementedError
    
    def gradient_sto(self, sentence, tag):
        raise NotImplementedError
    
    def predict_viterbi_tag_sequence(self):
        """
        """
        raise NotImplementedError

    def predict_posterior_tag_sequence(self):
        raise NotImplementedError

    def get_perplexity(self, sentence):
        return self.score_so(sentence)

    def generate_word_tag_sequence(self):
        raise NotImplementedError
    
    def update_parameter(self, new_param):
        """Change the 
        """
        raise NotImplementedError
    
    def get_copy_of_param(self):
        """This function returns me a COPY of the current params.
        It includes everything. The params used in the lm,
        the params used for state transitions of the tagging model.
        EVERYTHING!
        """
        raise NotImplementedError

    def update_parameters(self, delta):
        """A simple utility function for updating the parameters
        """
        raise NotImplementedError
