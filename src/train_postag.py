import sys
from parameter import Parameter
from hnmm import hnmm_full_observed_likelihood, hnmm_only_word_observed_likelihood
from gmemm import gmemm_full_observed_likelihood, gmemm_only_word_observed_likelihood
from util_misc import word_vocab_and_embedding, get_input_iterator
model_type=sys.argv[1] # HNMM, GMEMM, Simul
objective_type=sys.argv[2] # LL, NCE
optimization_type=sys.argv[3] # LBFGS, EM, Natural Gradient
word_vocab_file=open(sys.argv[4], "rb")
tag_vocab_file=open(sys.argv[5], "rb")
supervised_word_file=open(sys.argv[6], "rb")
supervised_tag_file=open(sys.argv[7], "rb")
unsupervised_word_file=open(sys.argv[8], "rb")
model_save_file=open(sys.argv[9], "wb")
unsup_ll_factor=float(sys.argv[10])
regularization_factor=float(sys.argv[11])
regularization_type = sys.argv[12]
[word_vocab, word_embedding]=word_vocab_and_embedding(word_vocab_file)
tag_vocab=[e.strip() for e in tag_vocab_file]
sup_word=list(get_input_iterator(supervised_word_file))
sup_tag=list(get_input_iterator(supervised_tag_file))
unsup_word=list(get_input_iterator(unsupervised_word_file))
parameter=Parameter(word_vocab, word_embedding, pos_vocab, unsup_ll_factor,
                    regularization_factor, sup_word, sup_tag,
                    regularization_type)
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
            ret+=hnmm_full_observed_likelihood(sentence, tag, parameter)
    elif model_type=="GMEMM":
        for sentence, tag in itertools.izip(sup_word, sup_tag):
            ret+=gmemm_full_observed_likelihood(sentence, tag, parameter)
    else:
        throw NotImplementedError
    return ret

def ll_unsup(unsup_word, parameter, model_type):
    ret=0.0
    if model_type=="HNMM":
        for sentence in unsup_word:
            ret+=hnmm_only_word_observed_likelihood(sentence, parameter)
    elif model_type=="GMEMM":
        for sentence in unsup_word:
            ret+=gmemm_only_word_observed_likelihood(sentence, parameter)
    else:
        throw NotImplementedError
    return ret

################################################################
## Use AD to calculate its gradient
## TODO: Test that AD and numerical diff give the same result.
################################################################

########################################
## Optimize the parameters using LBFGS
########################################

