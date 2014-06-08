def hnmm_only_word_observed_ll(word, parameter):
    """Use dynamic programming(specifically Variable Elimination)
    trick to marginalize over the unobserved tags
    Its complexity is sentence_length * (tags)^2"""
    phi=[]
    for factor_idx=xrange(0, len(word)-1):
        phi[factor_idx]={}
        for ti in parameter.tag_vocab:
            arr=[parameter.get_lp_t_given_w(ti, tim1, word[factor_idx])
                 + parameter.get_lp_w_given_two_tag_and_word_seq( \
                        word[factor_idx+1], ti, tim1, word[0:factor_idx+1])
                 for tim1 in parameter.tag_vocab]
            if factor_idx==0:
                arr1=[parameter.get_lp_w_given_t(word[0], t0)
                      + parameter.get_lp_t_given_w( \
                        t0, parameter.NULLTAG, parameter.BOS)
                      for t0 in parameter.tag_vocab]
            else:
                arr1=[phi[factor_idx-1][tim1]
                  for tim1 in parameter.tag_vocab]
                if factor_idx==len(word)-1:
                    arr2=[parameter.get_lp_w_given_two_tag_and_word_seq( \
                            parameter.EOS, ti, tim1, word)
                          for tim1 in parameter.tag_vocab]
                    arr1=list_add(arr1, arr2)
            phi[factor_idx][ti]=log_sum_exp(list_add(arr, arr1))
    return log_sum_exp(phi[factor_idx].values())

def hnmm_full_observed_ll(word, tag, parameter):
    """Return the loglikelihood under the HNMM model of a sequence of
    word, tag pairs. This is the simplest case for computation"""
    s=0.0
    s+=parameter.get_lp_t_given_w(tag[0], parameter.NULLTAG, parameter.BOS)
    s+=parameter.get_lp_w_given_t(word[0], tag[0])
    for i in xrange(1, len(tag)):
        s+=parameter.get_lp_t_given_w(tag[i], tag[i-1], word[i-1])
        s+=parameter.get_lp_w_given_two_tag_and_word_seq( \
            word[i], tag[i], tag[i-1], word[:i])
    s+=parameter.get_lp_w_given_two_tag_and_word_seq( \
        parameter.EOS, tag[-1], tag[-2], word)
    return s


