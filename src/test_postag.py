import sys, cPickle,
from hnmm import hnmm_tag_word
from gmemm import gmemm_tag_word
from parameter import Parameter
model_type=sys.argv[1]
model_file=open(sys.argv[2], "rb")
test_word_file=open(sys.argv[3], "rb")
predicted_tag_filename=sys.argv[4]
parameter=cPickle.load(model_file)
if model_type=="HNMM":
    for word in get_input_iterator(test_word_file):
        print hnmm_tag_word(word, parameter)
elif model_type=="GMEMM":
    for word in get_input_iterator(test_word_file):
        print gmemm_tag_word(word, parameter)
else:
    throw NotImplementedError
