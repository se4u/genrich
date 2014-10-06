import sys
import itertools
import numpy as np
from math import log, exp
from parameter import Parameter
from hnmm import hnmm_full_observed_ll, hnmm_only_word_observed_ll
from gmemm import gmemm_full_observed_ll, gmemm_only_word_observed_ll
from util_oneliner import get_vocab_from_file
from util_datamunge import word_vocab_and_embedding
import optparse
def get_param(l, k):
    "Get sys.argv as l and get the value of the corresponding option"

options=dict(e.split("=") for e in sys.argv[1:])
model_type        = options["MODEL_TYPE"]
cpd_type          = options["CPD_TYPE"]
objective_type    = options["OBJECTIVE_TYPE"]
optimization_type = options["OPTIMIZATION_TYPE"]
unsup_ll_weight   = float(options["UNSUP_LL_WEIGHT"])
param_reg_type    = options["PARAM_REG_TYPE"]
param_reg_weight  = options["PARAM_REG_WEIGHT"]
tag_vocab         = get_vocab_from_file(open(options["TAG_VOCAB_FILE"], "rb"))
word_vocab        = get_vocab_from_file(open(options["WORD_VOCAB_FILE"], "rb"))
sup_train_file    = open(options["SUP_TRAIN_FILE"], "rb")
unsup_train_file  = open(options["UNSUP_TRAIN_FILE"], "rb")
word_embedding_filename = options["WORD_EMBEDDING_FILE"]
save_filename     = options["SAVE_FILE"]

wordtag=token_iterator(sup_train_file)

# unsup_word=list(get_input_iterator(unsupervised_word_file))
# bilinear_init_sigma=0.01
# t_given_w_lambda=0.1
# w_given_t_BOS_lambda=0.01

parameter=Parameter(word_vocab,
                    word_embedding,
                    EOS_embedding,
                    tag_vocab,
                    unsup_ll_factor,
                    regularization_factor,
                    sup_word,
                    sup_tag,
                    regularization_type,
                    bilinear_init_sigma=bilinear_init_sigma,
                    t_given_w_lambda=t_given_w_lambda,
                    w_given_t_BOS_lambda=w_given_t_BOS_lambda)
if model_save_filename.startswith(r"res/postag_small.model"):
    assert word_vocab==["A", "B", "C", "D", "E"]
    assert parameter.get_word_idx("E")==5
    assert parameter.get_tag_idx("4")==4
    assert parameter.R==5
    assert all(parameter.RW[:, 5]==[0, 0, 0, 0, 1])
    assert close(parameter.get_lp_t_given_w("1", parameter.NULLTAG, parameter.BOS), log((3+t_given_w_lambda)/(4+t_given_w_lambda*4)))
    #print close(parameter.get_lp_t_given_w("3", "1", "E"), log(t_given_w_lambda/(1+t_given_w_lambda*4)))
    assert parameter.get_lp_t_given_w("3", "4", "E")==0.0
    assert close(parameter.get_lp_w_given_t_BOS("A", "1"), log((3+w_given_t_BOS_lambda)/(3+w_given_t_BOS_lambda*5)))
    assert parameter.get_lp_w_given_two_tag_and_word_seq(parameter.EOS, "3", parameter.NULLTAG, ["E"], [0]*5)
    rabc = np.dot(parameter.C1, parameter.RW[:,3]) + np.dot(parameter.C2, np.dot(parameter.C1, parameter.RW[:,2])) + np.dot(parameter.C2, np.dot(parameter.C2, np.dot(parameter.C1, parameter.RW[:,1])))
    assert close(rabc, parameter.get_next_seq_embedding(["A", "B", "C"], parameter.get_next_seq_embedding(["A", "B"], parameter.get_next_seq_embedding(["A"], None))))
    

    
#####################################################
# DEFINE Log Likelihood Functions for HNMM and GMEMM 
def ll(sup_word, sup_tag, unsup_word, parameter, model_type):
    ret=ll_sup(sup_word, sup_tag, parameter, model_type)
    ret+= parameter.unsup_ll_factor*ll_unsup(unsup_word, parameter, model_type)
    ret+=parameter.regularization_factor*parameter.regularization_contrib()
    return ret

def ll_sup(sup_word, sup_tag, parameter, model_type):
    if model_type=="HNMM":
        arr=[hnmm_full_observed_ll(sentence, tag, parameter)
             for sentence, tag
             in itertools.izip(sup_word, sup_tag)]
    elif model_type=="GMEMM":
        arr=[gmemm_full_observed_ll(sentence, tag, parameter)
             for sentence, tag
             in itertools.izip(sup_word, sup_tag)]
    else:
        raise NotImplementedError
    return sum(arr)

def ll_unsup(unsup_word, parameter, model_type):
    if model_type=="HNMM":
        arr=[hnmm_only_word_observed_ll(sentence, parameter)
             for sentence in unsup_word]
    elif model_type=="GMEMM":
        arr=[gmemm_only_word_observed_ll(sentence, parameter)
             for sentence in unsup_word]
    else:
        raise NotImplementedError
    return sum(arr)

if model_save_filename.startswith(r"res/postag_small.model"):
    # E 3 is correct
    ar= [parameter.get_lp_w_given_two_tag_and_word_seq(\
            parameter.EOS, "3", parameter.NULLTAG, ["E"], 
            parameter.get_next_seq_embedding(["E"], None)),
         parameter.get_lp_t_given_w("3", parameter.NULLTAG, parameter.BOS),
         parameter.get_lp_w_given_t_BOS("E", "3")]
    assert close(sum(ar), ll_sup(["E"], ["3"], parameter, "HNMM"))
    # unsupervised C is correct
    ar=[ll_sup(["C"], [str(i)], parameter, "HNMM") for i in range(1,5)]
    assert close(log_sum_exp(ar), ll_unsup([["C"]], parameter, "HNMM"))
    # Supervised A B 1 2
    ar = [parameter.get_lp_w_given_t_BOS("A", "1"),
          parameter.get_lp_t_given_w("1", parameter.NULLTAG, parameter.BOS),
          parameter.get_lp_t_given_w("2", "1", "A"),
          parameter.get_lp_w_given_two_tag_and_word_seq(\
            "B", "2", "1", ["A"],
            parameter.get_next_seq_embedding(["A"], None)),
          parameter.get_lp_w_given_two_tag_and_word_seq(\
            parameter.EOS, "2", "1", ["A", "B"], 
            parameter.get_next_seq_embedding(
                ["A", "B"],
                parameter.get_next_seq_embedding(
                    ["A"], None)))]
    assert close(sum(ar), ll_sup([["A", "B"]], [["1", "2"]], parameter, "HNMM"))
    # UNsupervised A D 
    ar=[ll_sup([["A", "D"]], [[str(i), str(j)]], parameter, "HNMM") for i in range(1,5) for j in range(1,5)]
    assert close(log_sum_exp(ar), ll_unsup([["A", "D"]], parameter, "HNMM"))
    print ll(sup_word, sup_tag, unsup_word, parameter, model_type)
    pass

################################################################
## Use AD to calculate its gradient
## TODO: Test that AD and numerical diff give the same result.
################################################################

########################################
## Optimize the parameters using LBFGS
########################################

########################################
## Save the parameter object to file
########################################
parameter.serialize(model_save_filename)
