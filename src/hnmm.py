from misc_util import log_sum_exp, list_add
def hnmm_only_word_observed_ll(word, parameter):
    """Use dynamic programming(specifically Variable Elimination)
    trick to marginalize over the unobserved tags
    Its complexity is sentence_length * (tags)^2"""
    if len(word)==1:
        factor_idx=0
        phi=[{}]
        nse=parameter.get_next_seq_embedding(word,None)
        for ti in parameter.tag_vocab:
            phi[factor_idx][ti]= \
                parameter.get_lp_t_given_w(ti, parameter.NULLTAG,
                                           parameter.BOS) \
                + parameter.get_lp_w_given_t_BOS(word[0], ti) \
                + parameter.get_lp_w_given_two_tag_and_word_seq(
                                               parameter.EOS,
                                               ti,
                                               parameter.NULLTAG,
                                               word,
                                               nse)
    else:
        ###########################################################
        ## The General case applies to sentence of length >= 2
        ###########################################################
        phi=[None]*(len(word)-1)
        seq_embedding=None
        for factor_idx in xrange(len(word)-1):
            phi[factor_idx]={}
            prev_word_seq=word[0:factor_idx+1]
            seq_embedding=parameter.get_next_seq_embedding(prev_word_seq,
                                                 seq_embedding)
            for ti in parameter.tag_vocab:
                arr=[parameter.get_lp_t_given_w(ti, tim1, word[factor_idx])
                     + parameter.get_lp_w_given_two_tag_and_word_seq( \
                        word[factor_idx+1], ti, tim1, prev_word_seq,
                        seq_embedding)
                     for tim1 in parameter.tag_vocab]
                if factor_idx==0:
                    arr1=[parameter.get_lp_w_given_t_BOS(word[0], t0)
                          + parameter.get_lp_t_given_w( \
                            t0, parameter.NULLTAG, parameter.BOS)
                          for t0 in parameter.tag_vocab]
                else:
                    arr1=[phi[factor_idx-1][tim1]
                          for tim1 in parameter.tag_vocab]
                    pass
                #######################################################
                #### This SPECIAL CASE TO ADD EOS TERM AT THE VERY END
                ######################################################
                if factor_idx==len(word)-2:
                    prev_word_seq=word[0:factor_idx+2]
                    EOS_seq_embedding=parameter.get_next_seq_embedding(\
                        prev_word_seq,seq_embedding)
                    arr2=[parameter.get_lp_w_given_two_tag_and_word_seq( \
                            parameter.EOS, ti, tim1, word, EOS_seq_embedding)
                          for tim1 in parameter.tag_vocab]
                    arr1=list_add(arr1, arr2)
                phi[factor_idx][ti]=log_sum_exp(list_add(arr, arr1))
    return log_sum_exp(phi[factor_idx].values())

def hnmm_full_observed_ll(word, tag, parameter):
    """Return the loglikelihood under the HNMM model of a sequence of
    word, tag pairs. This is the simplest case for computation"""
    s=0.0
    s+=parameter.get_lp_t_given_w(tag[0], parameter.NULLTAG, parameter.BOS)
    s+=parameter.get_lp_w_given_t_BOS(word[0], tag[0])
    seq_embedding=None
    for i in xrange(1, len(tag)):
        s+=parameter.get_lp_t_given_w(tag[i], tag[i-1], word[i-1])
        prev_word_seq=word[:i]
        seq_embedding=parameter.get_next_seq_embedding(prev_word_seq, seq_embedding)
        s+=parameter.get_lp_w_given_two_tag_and_word_seq( \
            word[i], tag[i], tag[i-1], prev_word_seq, seq_embedding)
    prev_word_seq=word
    seq_embedding=parameter.get_next_seq_embedding(prev_word_seq, seq_embedding)
    ppt = parameter.NULLTAG if len(tag) == 1 else tag[-2]
    s+=parameter.get_lp_w_given_two_tag_and_word_seq( \
        parameter.EOS, tag[-1], ppt, word, seq_embedding)
    return s



