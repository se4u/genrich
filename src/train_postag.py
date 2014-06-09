import sys
import itertools
from math import log, exp
from parameter import Parameter
from hnmm import hnmm_full_observed_ll, hnmm_only_word_observed_ll
from gmemm import gmemm_full_observed_ll, gmemm_only_word_observed_ll
from misc_util import word_vocab_and_embedding, get_input_iterator, close
model_type=sys.argv[1] # HNMM, GMEMM, Simul
objective_type=sys.argv[2] # LL, NCE
optimization_type=sys.argv[3] # LBFGS, EM, Natural Gradient
word_vocab_file=open(sys.argv[4], "rb")
tag_vocab_file=open(sys.argv[5], "rb")
supervised_word_file=open(sys.argv[6], "rb")
supervised_tag_file=open(sys.argv[7], "rb")
unsupervised_word_file=open(sys.argv[8], "rb")
model_save_filename=sys.argv[9]
unsup_ll_factor=float(sys.argv[10])
regularization_factor=float(sys.argv[11])
regularization_type = sys.argv[12]
[word_vocab, word_embedding]=word_vocab_and_embedding(word_vocab_file)
tag_vocab=[e.strip() for e in tag_vocab_file]
sup_word=list(get_input_iterator(supervised_word_file))
sup_tag=list(get_input_iterator(supervised_tag_file))
unsup_word=list(get_input_iterator(unsupervised_word_file))
bilinear_init_sigma=0.01
t_given_w_lambda=0.1
w_given_t_BOS_lambda=0.01
parameter=Parameter(word_vocab, word_embedding, tag_vocab, unsup_ll_factor,
                    regularization_factor, sup_word, sup_tag,
                    regularization_type,
                    bilinear_init_sigma=bilinear_init_sigma,
                    t_given_w_lambda=t_given_w_lambda,
                    w_given_t_BOS_lambda=w_given_t_BOS_lambda)
if model_save_filename.startswith(r"res/postag_small.model"):
    print word_vocab==["A", "B", "C", "D", "E"]
    print parameter.get_word_idx("E")==5
    print parameter.get_tag_idx("4")==4
    print parameter.R==10
    print parameter.RW[:, 5]==[0, 0, 0, 0, 1, 1, 0, 1, 1, 0,]
    print close(parameter.get_lp_t_given_w("1", parameter.NULLTAG, parameter.BOS), log((3+t_given_w_lambda)/(4+t_given_w_lambda*4)))
    print close(parameter.get_lp_t_given_w("3", "1", "E"), log(t_given_w_lambda/(1+t_given_w_lambda*4)))
    print parameter.get_lp_t_given_w("3", "4", "E")==0.0
    print close(parameter.get_lp_w_given_t_BOS("A", "1"), log((3+w_given_t_BOS_lambda)/(3+w_given_t_BOS_lambda*5)))
    print parameter.get_lp_w_given_two_tag_and_word_seq(parameter.EOS, "3", parameter.NULLTAG, ["E"], [0]*10)
    
########################################################
## DEFINE Log Likelihood Functions for HNMM and GMEMM 
########################################################
def ll(sup_word, sup_tag, unsup_word, parameter, model_type):
    ret=ll_sup(sup_word, sup_tag, parameter, model_type)
    ret+= parameter.unsup_ll_factor*ll_unsup(unsup_word, parameter, model_type)
    ret+=parameter.regularization_factor*parameter.regularization_contrib()
    return ret

def ll_sup(sup_word, sup_tag, parameter, model_type):
    ret=0.0
    if model_type=="HNMM":
        for sentence, tag in itertools.izip(sup_word, sup_tag):
            ret+=hnmm_full_observed_ll(sentence, tag, parameter)
    elif model_type=="GMEMM":
        for sentence, tag in itertools.izip(sup_word, sup_tag):
            ret+=gmemm_full_observed_ll(sentence, tag, parameter)
    else:
        raise NotImplementedError
    return ret

def ll_unsup(unsup_word, parameter, model_type):
    ret=0.0
    if model_type=="HNMM":
        for sentence in unsup_word:
            ret+=hnmm_only_word_observed_ll(sentence, parameter)
    elif model_type=="GMEMM":
        for sentence in unsup_word:
            ret+=gmemm_only_word_observed_ll(sentence, parameter)
    else:
        raise NotImplementedError
    return ret

print "Initial ll = ", ll(sup_word, sup_tag, unsup_word, parameter, model_type)
################################################################
## Use AD to calculate its gradient
## TODO: Test that AD and numerical diff give the same result.
################################################################

########################################
## Optimize the parameters using LBFGS
########################################


