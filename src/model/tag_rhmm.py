
class tag_rhmm(object):
    def __init__(self, cpd_type, objective_type, param_reg_type,
                  param_reg_weight,
                  unsup_ll_weight,
                  word_embedding_filename,
                  initialize_embedding_according_to_file,
                  initialize_param_to_random):
        """The cpd_type defines the type of cpd to use, including the
        smoothing and the model, hinton's lbl or structured sparse stuff.
        """
        pass
    
    def score_sto(self, sentence, tag_possibly_none):
        """sto means sentence and tag observed.
        but here the tags can be missing and represented by none in
        which case we need to do the DP. This is the general case.
        """
        pass
    
    def score_ao(self, sentence, tag):
        """ao means all observed. That both the tag and the word were observed
        """
        pass

    def score_so(self, sentence):
        """so means that the sentence was observed, but not the tags
        This requires the use of DP. We are calculating the
        E[P(sentence)] given the parameters.
        This only needs a forward pass to compute the total probability.
        """
        pass

    def gradient_so(self, sentence):
        pass
    
    def gradient_sto(self, sentence, tag):
        pass
    
    def predict_viterbi_tag_sequence(self):
        """
        """
        pass

    def predict_posterior_tag_sequence(self):
        pass

    def get_perplexity(self, sentence):
        return self.score_so(sentence)

    def generate_word_tag_sequence(self):
        pass
    def update_parameter(self, new_param):
        """Change the 
        """
        pass
    def get_copy_of_param(self):
        """This function returns me a COPY of the current params.
        It includes everything. The params used in the lm,
        the params used for state transitions of the tagging model.
        EVERYTHING!
        """
        pass
    
def hnmm_only_word_observed_ll(word, parameter):
    """Use dynamic programming
    (specifically Variable Elimination)
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



def hnmm_tag_word(word, parameter):
    """ Given a sentence of word predict its tags.
    parameter object holds all the relevant information
    Return a list of three things
    [word likelihood, predicted tag, tag probabilities]
    """
    
    pass
