"""The base module for all taggers. This base class is subclassed by
the order0hmm tagger as well as the more baroque taggers.
"""
# Created: 13 October 2014
__author__  = "Pushpendre Rastogi"
__contact__ = "pushpendre@jhu.edu"

class tag_baseclass(object):
    """This class decsribes all the functionalities that a model would
    need to have. An important thing to note is that its score must
    always be defined in a way so that minimizing it gives the right
    answer.  There is no other programatic way to enforce this.
    """
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
        smoothing and the model, hinton's lbl or structured sparse stuff.
        """
        raise NotImplementedError
    
    def score_sto(self, sentence, tag_possibly_none):
        """sto means sentence and tag observed.
        but here the tags can be missing and represented by none in
        which case we need to do the DP. This is the general case.
        """
        raise NotImplementedError
    
    def score_ao(self, sentence, tag):
        """ao means all observed. That both the tag and the word were observed
        """
        raise NotImplementedError

    def score_so(self, sentence):
        """so means that the sentence was observed, but not the tags
        This requires the use of DP. We are calculating the
        E[P(sentence)] given the parameters.
        This only needs a forward pass to compute the total probability.
        """
        raise NotImplementedError

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
