"""An implementation of the rhmm model. Which leverages previous two
tags and k previous words while assiging a tag to the current word. 
"""
# Created: 23 November 2014
__author__  = "Pushpendre Rastogi"
__contact__ = "pushpendre@jhu.edu"
import logging, sys
import util_oneliner
from util_theano_extension import log_softmax
from tag_order0hmm import tag_order0hmm
import numpy as np
import theano.tensor as T
from theano import function, scan, shared
import yaml
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# emb means embedding
# App Hungarian
#   tg_ means theano graph object
#   tf_ means theano function object
#   na_ means numpy array
#   nam_ means numpy array representing a transforming matrix
#   nat_ means numpy array, a 3d tensor with a number of transformation matrices
def lw_s(tg_word_idx,
         tg_tagh_emb1, tg_S_emb,
         tg_word_emb, tg_Ttld1, tg_Ttld2):
    """ Find the l(w_1 | T_1, S). The log probability of
    first word given start tag and first tag's embedding
    tg_word_idx is the row_id of the word in the word_embedding matrix
    tg_tagh_emb1, tg_S_emb are the previous tags embeddings.
    The last 3 parameters are the embedding data srtuctures.
    """
    return log_softmax(tg_word_idx,
                       T.dot(tg_word_emb, 
                             T.dot(tg_Ttld1, tg_tagh_emb1)+
                             T.dot(tg_Ttld2, tg_S_emb))
                       )

def lt_s(tg_tag_idx,
         tg_S_emb,
         tg_tag_emb, tg_T1):
    """ Find l(T_1 | S). The log probability of first
    tag given start tag.
    """
    return lw_s(tg_tag_idx,
                tg_S_emb, 0,
                tg_tag_emb, tg_T1, 0)

def lt_rest(tg_tag_idx, tg_tagh_emb1, tg_tagh_emb2, tg_wordh_idx_arr,
            tg_tag_emb, tg_word_emb, tg_T1, tg_T2, tg_W):
    """ Find probability of the tag keyed by tg_tag_idx given the history that
    is specified by tagh_emb[12] and wordh_idx_arr
    The rest of the arguments are parameters of the model.
    All the embeddings contain the embeddings/features of words/tags in a row.
    """
    _transform_wh_emb_using_corresponding_W_mat=\
        lambda i, w_idx: T.dot(tg_W[i], tg_word_emb[w_idx,:])
    return log_softmax(tg_tag_idx,
                       T.dot(tg_tag_emb,
                             T.dot(tg_T1, tg_tagh_emb1) +
                             T.dot(tg_T2, tg_tagh_emb2) +
                             scan(_transform_wh_emb_using_corresponding_W_mat,
                                  sequences=[T.arange(tg_wordh_idx_arr.shape[0]),
                                             tg_wordh_idx_arr
                                             ]
                                  )[0].sum(0)
                             )
                       )

def lw_rest(tg_word_idx, tg_tagh_emb1, tg_tagh_emb2, tg_wordh_idx_arr,
            tg_word_emb, tg_Ttld1, tg_Ttld2, tg_Wtld):
    """ Find probability of the word keyed by word_idx.
    The context is specified by tagh_emb[12] and wordh_idx_arr
    I do not need to know tg_tag_emb since all the tag embeddings
    are explicitly provided anyway.
    """
    return lt_rest(tg_word_idx, tg_tagh_emb1, tg_tagh_emb2, tg_wordh_idx_arr,
                   tg_word_emb, tg_word_emb,
                   tg_Ttld1, tg_Ttld2, tg_Wtld)

class tag_rhmm(tag_order0hmm):
    """ The rich hmm class.
    It basically conditions on a lot of word/tag features.
    """
    def __init__(self,
                 word_context_size,
                 embedding_size,
                 objective_type,
                 param_reg_type,
                 param_reg_weight,
                 init_from_file,
                 param_filename,
                 tag_vocab,
                 word_vocab,
                 test_time=True):
        if init_from_file:
            data=yaml.load(open(param_filename, "rb"))
            self.word_context_size=data["word_context_size"]
            self.embedding_size=data["embedding_size"]
            self.objective_type=data["objective_type"]
            self.param_reg_type=data["param_reg_type"]
            self.param_reg_weight=data["param_reg_weight"]
            self.tag_vocab=data["tag_vocab"]
            self.word_vocab=data["word_vocab"]
            self.num_tag=len(self.tag_vocab)
            self.num_word=len(self.word_vocab)
            # Following are parameters of the model.
            # These need to be instantiated.
            laff = lambda nm : np.loads(data[nm])
            # laff means load arr from file. It takes (nm) name of variable
            self.na_tag_emb=laff("tag_emb")
            self.nv_S_emb=laff("S_emb")
            self.na_word_emb=laff("word_emb")
            self.nam_T1=laff("nam_T1")
            self.nam_T2=laff("nam_T2")
            self.nam_Ttld1=laff("nam_Ttld1")
            self.nam_Ttld2=laff("nam_Ttld2")
            self.nat_W=laff("nat_W")
            self.nat_Wtld=laff("nat_Wtld")
        else:
            assert("E" not in tag_vocab)
            tag_vocab["E"]=len(tag_vocab)
            self.word_context_size=word_context_size
            self.embedding_size=embedding_size
            self.objective_type=objective_type
            self.param_reg_type=param_reg_type
            self.param_reg_weight=param_reg_weight
            self.num_tag=len(tag_vocab)
            self.num_word=len(word_vocab)
            self.tag_vocab=tag_vocab
            self.word_vocab=word_vocab
            # Following are parameters of the model.
            # They are randomly instantiated.
            es=self.embedding_size
            make_T = lambda : np.random.randn(es,es)/pow(es,0.5)
            make_W = lambda : np.random.randn(self.word_context_size,
                                              es,es)/pow(es,0.5)
            # Added 2 to accomodate embeddings (E) End tag
            self.na_S_emb=util_oneliner.get_random_emb(1,es)
            self.na_tag_emb=util_oneliner.get_random_emb(self.num_tag,es)
            self.na_word_emb=util_oneliner.get_random_emb(self.num_word,es)
            self.nam_T1=make_T()
            self.nam_T2=make_T()
            self.nam_Ttld1=make_T()
            self.nam_Ttld2=make_T()
            self.nat_W=make_W()
            self.nat_Wtld=make_W()
        self.tg_S_emb=shared(self.na_S_emb, borrow=True)
        self.tg_tag_emb=shared(self.na_tag_emb, borrow=True)
        self.tg_word_emb=shared(self.na_word_emb, borrow=True)
        self.tg_T1=shared(self.nam_T1, borrow=True)
        self.tg_T2=shared(self.nam_T2, borrow=True)
        self.tg_Ttld1=shared(self.nam_Ttld1, borrow=True)
        self.tg_Ttld2=shared(self.nam_Ttld2, borrow=True)
        self.tg_W=shared(self.nat_W, borrow=True)
        self.tg_Wtld=shared(self.nat_Wtld, borrow=True)
        self.params=[self.tg_S_emb, self.tg_tag_emb, self.tg_word_emb,
                     self.tg_T1, self.tg_T2,
                     self.tg_Ttld1, self.tg_Ttld2,
                     self.tg_W, self.tg_Wtld]
        # Now create theano graph of log likelihood of supervised sequence
        tag_ids=T.ivector("tag_ids")
        word_ids=T.ivector("word_ids")
        if self.objective_type=="LL":
            # Compute the scores
            self._tg_score_ao=self.make_tg_score_ao(tag_ids,word_ids)+\
                self.get_tg_penalty(test_time)
            # Compute the gradient
            self._tg_gradient_ao=T.grad(self._tg_score_ao,self.params)
            # Following are compiled functions.
            self._score_ao=function([tag_ids, word_ids], self._tg_score_ao,
                                    name="_score_ao")
            self._gradient_ao=function([tag_ids, word_ids],self._tg_gradient_ao,
                                       name="_gradient_ao")
            eta=T.fscalar("eta")
            self._update_ao=function([eta, tag_ids, word_ids],
                                     self._tg_gradient_ao,
                                     name="update_ao",
                                     updates=[(p, p+eta*g)
                                              for (g, p)
                                              in zip(self._tg_gradient_ao,
                                                     self.params)])
            if test_time:
                tg_score_so=self.make_tg_score_so(word_ids)
                self._score_so=function([word_ids], tg_score_so,
                                        name="_score_so")
            else:
                pass
        else:
            raise NotImplementedError(
                    "objective_type: %s"%self.cpd_type)
        return
    
    def make_tg_score_ao(self, tag_ids, word_ids):
        """ The last tag should be the End tag
        """
        def _get_lp_rest_of_sentence(i, word_context_size,
                                     tg_tag_emb, tg_word_emb,
                                     tg_T1, tg_T2, tg_W,
                                     tg_Ttld1, tg_Ttld2, tg_Wtld):
            _word, _ = scan(lambda x: word_ids[x],
                            sequences=\
                                T.arange(i,
                                         T.maximum(-1, i-word_context_size),
                                         -1)
                            )
            tg_tagh_emb1 = tg_tag_emb[tag_ids[i],:].flatten()
            tg_tagh_emb2 = tg_tag_emb[tag_ids[i-1],:].flatten()
            p_lt = lt_rest(tg_tag_idx = tag_ids[i+1],
                           tg_tagh_emb1 = tg_tagh_emb1,
                           tg_tagh_emb2 = tg_tagh_emb2,
                           tg_wordh_idx_arr = _word,
                           tg_tag_emb = tg_tag_emb,
                           tg_word_emb = tg_word_emb,
                           tg_T1 = tg_T1,
                           tg_T2 = tg_T2,
                           tg_W = tg_W)
            p_lw = lw_rest(tg_word_idx = word_ids[i],
                           tg_tagh_emb1 = tg_tagh_emb1,
                           tg_tagh_emb2 = tg_tagh_emb2,
                           tg_wordh_idx_arr = _word[1:],
                           #tg_tag_emb = tg_tag_emb,
                           tg_word_emb = tg_word_emb,
                           tg_Ttld1 = tg_Ttld1,
                           tg_Ttld2 = tg_Ttld2,
                           tg_Wtld = tg_Wtld)
            return p_lt + p_lw
        
        pt1_s = lt_s(tg_tag_idx=tag_ids[0],
                     tg_S_emb=self.tg_S_emb.flatten(),
                     tg_tag_emb=self.tg_tag_emb,
                     tg_T1=self.tg_T1)
        pw1_s = lw_s(tg_word_idx=word_ids[0],
                     tg_tagh_emb1=self.tg_tag_emb[tag_ids[0],:].flatten(),
                     tg_S_emb=self.tg_S_emb.flatten(),
                     tg_word_emb=self.tg_word_emb,
                     tg_Ttld1=self.tg_Ttld1,
                     tg_Ttld2=self.tg_Ttld2)
        pt2_t1sw = lt_rest(tg_tag_idx = tag_ids[1],
                           tg_tagh_emb1 = \
                               self.tg_tag_emb[tag_ids[0],:].flatten(),
                           tg_tagh_emb2 = self.tg_S_emb.flatten(),
                           tg_wordh_idx_arr = word_ids[0:1],
                           tg_tag_emb = self.tg_tag_emb,
                           tg_word_emb = self.tg_word_emb,
                           tg_T1 = self.tg_T1,
                           tg_T2 = self.tg_T2,
                           tg_W = self.tg_W)
        p_rest, _ = scan(_get_lp_rest_of_sentence,
                         sequences=T.arange(1,word_ids.shape[0]),
                         non_sequences=[self.word_context_size,
                                        self.tg_tag_emb,
                                        self.tg_word_emb,
                                        self.tg_T1,
                                        self.tg_T2,
                                        self.tg_W,
                                        self.tg_Ttld1,
                                        self.tg_Ttld2,
                                        self.tg_Wtld])
        return pw1_s + pt1_s + pt2_t1sw + p_rest.sum()

    def score_ao(self, tags, words):
        """This function receives actual words and tags (not indices!) 
        and then returns their probabilties.
        """
        assert len(tags) > 2
        assert len(tags)==len(words)
        return self._score_ao(
            [self.get_from_tag_vocab(t) for t in tags+["E"]],
            [self.get_from_word_vocab(w) for w in words]
            )
    
    def make_tg_score_so(self, word_ids):
        pass
    
    def predict_posterior_tag_sequence(self, words):
        pass
    
    def predict_viterbi_tag_sequence(self, words):
        pass

    def get_dict_for_save(self):
        return dict(word_context_size=self.word_context_size,
                    embedding_size=self.embedding_size,
                    objective_type=self.objective_type,
                    param_reg_type=self.param_reg_type,
                    param_reg_weight=self.param_reg_weight,
                    tag_vocab=self.tag_vocab,
                    word_vocab=self.word_vocab,
                    na_tag_emb=self.na_tag_emb.dumps(),
                    na_word_emb=self.na_word_emb.dumps(),
                    nam_T1=self.nam_T1.dumps(),
                    nam_T2=self.nam_T2.dumps(),
                    nam_Ttld1=self.nam_Ttld1.dumps(),
                    nam_Ttld2=self.nam_Ttld2.dumps(),
                    nat_W=self.nat_W.dumps(),
                    nat_Wtld=self.nat_Wtld.dumps())
    
    
