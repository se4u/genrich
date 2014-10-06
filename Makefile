.PHONY: postagger_accuracy_small res/postag_small.model log/postag_small.pos.test.predict
# .SECONDARY: 
PYCMD := python
RP := res/postag
MODELTYPE := HNMM
OBJECTIVE_TYPE := LL
OPTIMIZATION_TYPE := LBFGS
UNSUP_LL_FACTOR := 1.0
REGULARIZATION_FACTOR := 0.1
REGULARIZATION_TYPE := l2

# Perform training and testing using the small test data on 3 different types of models.
all: # Train_train > Predict_train > Evaluate_train > Tune.parameters + Profile/debug.code (Loop)  > Train_train > Predict_dev
	make -n postag_accuracy_small.GMEMM MODELTYPE=GMEMM && \
	make -n postag_accuracy_small.HNMM MODELTYPE=HNMM && \
	make -n postag_accuracy_small.SIMUL MODELTYPE=SIMUL

#####################################
### EVALUATION
# TARGET : Printed output of accuracy.
# SOURCE : 1. The predicted postags 2. The actual postags
postag_accuracy_%.$(MODELTYPE) : log/postag_%.pos.test.predict.$(MODELTYPE) res/postag_%.pos.test
	$(PYCMD) src/postag_accuracy.py $+

####################################
## POSTAGGER PREDICTION 
# TARGET : The output of the postagger 
# SOURCE : The trained postagger model,
#	   The raw words that we want to postag using this model
log/postag_%.pos.test.predict.$(MODELTYPE) : $(RP)_%.model.$(MODELTYPE) $(RP)_%.word.test
	$(PYCMD) src/test_postag.py $(MODELTYPE) $+ $@

####################################
## POSTAGGER TRAINING
# TARGET: The postagger model file
# Example targets are res/postag_
# SOURCE: First two are vocabulary files. vocabulary of postagger and words
#         Next we have the supervised data
#	  Then we have the unsupervised data
# sample sources are the
# # Train_train > Predict_train > Evaluate_train > Tune.parameters + Profile/debug.code (Loop)  > Train_train
# Remember what's the basic idea.
# The basic idea is to condition on long contexts.
# It takes the following arguments
# MODEL_TYPE = tdgd bugp smgp # (Top down, Bottom up, Simultaneous) Generative PD
# CPD_TYPE   = lbl, addlambda<float>, unsmooth, chengoodman # Smoothing methods for the long context CPD
# OBJECTIVE = ll, nce # Log likelihood or NCE objective 
# OPTIMIZATION        = lbfgs, em, sgd, naturalgrad #LBFGS, EM, SGD, Natural Gradient
# UNSUP_LL_WEIGHT     = <float> # This is Multi conditional learning
# PARAM_REG_TYPE      = l2, l1  # L2 or L1 regularization over the priors
# PARAM_REG_WEIGHT    = <float> # The weight of the regularization term
# TAG_VOCAB_FILE      = name of the tag vocabulary file.
# WORD_VOCAB_FILE     = name of the word vocab file
# SUP_TRAIN_FILE      = name of the training file with supervised data
# UNSUP_TRAIN_FILE    = name of the training file with unsupervised data
# SAVE_FILE            = name of the output pickle of the trained file
TRAIN_OPT_EXTRACTOR = $(word $(subst ., ,$*),$1)
res/train_tag_% :
	WORD_VOCAB_FILE=res/$(call TRAIN_OPT_EXTRACTOR,8) TAG_VOCAB_FILE=res/$(call TRAIN_OPT_EXTRACTOR,9) SUP_TRAIN_FILE=res/$(call TRAIN_OPT_EXTRACTOR,10) UNSUP_TRAIN_FILE=res/$(call TRAIN_OPT_EXTRACTOR,11) WORD_EMBEDDING_FILE=res/$(call TRAIN_OPT_EXTRACTOR,12) \
	$(MAKE) MODEL_TYPE=$(call TRAIN_OPT_EXTRACTOR,1) \
	        CPD_TYPE=$(call TRAIN_OPT_EXTRACTOR,2)
	        OBJECTIVE_TYPE=$(call TRAIN_OPT_EXTRACTOR,3) \
	        OPTIMIZATION_TYPE=$(call TRAIN_OPT_EXTRACTOR,4) \
	        UNSUP_LL_WEIGHT=$(call TRAIN_OPT_EXTRACTOR,5) \
	        PARAM_REG_TYPE=$(call TRAIN_OPT_EXTRACTOR,6) \
	        PARAM_REG_WEIGHT=$(call TRAIN_OPT_EXTRACTOR,7) \
	        TAG_VOCAB_FILE=$$TAG_VOCAB_FILE \
	        WORD_VOCAB_FILE=$$VOCAB_FILE \
	        SUP_TRAIN_FILE=$$SUP_TRAIN_FILE \
	        UNSUP_TRAIN_FILE=$$UNSUP_TRAIN_FILE \
	        WORD_EMBEDDING_FILE=$$WORD_EMBEDDING_FILE \
	        SAVE_FILE=$@ \
	        MYDEP="$$VOCAB_FILE $$TRAIN_FILE $$SUP_TRAIN_FILE $$UNSUP_TRAIN_FILE" \
            train_tag_generic

train_tag_generic: $(MYDEP)
	$(PYCMD) src/train_tag.py MODEL_TYPE=$(MODELTYPE) CPD_TYPE=$(CPD_TYPE) OBJECTIVE_TYPE=$(OBJECTIVE) OPTIMIZATION_TYPE=$(OPTIMIZATION_TYPE) UNSUP_LL_WEIGHT=$(UNSUP_LL_WEIGHT) PARAM_REG_TYPE=$(PARAM_REG_TYPE) PARAM_REG_WEIGHT=$(PARAM_REG_WEIGHT) TAG_VOCAB_FILE=$(TAG_VOCAB_FILE) WORD_VOCAB_FILE=$(WORD_VOCAB_FILE) SUP_TRAIN_FILE=$(SUP_TRAIN_FILE) UNSUP_TRAIN_FILE=$(UNSUP_TRAIN_FILE) SAVE_FILE=$(SAVE_FILE)


#####################################
## BASELINE (CRFSuite)
# I initially thought of using CRF++, CRFSuite (sgd, Wapiti, Mallet
# etc. seemed too old) but CRFSuite had a python wrapper, and people
# seemed to be moving to it (It was cited a lot, so I decided to use
# only this)

# CRFSuite also has benchmarks on its site for the CONLL 200 chunking
# shared task which seemed a good enough approximation of postagging. 
# They also have pos tagging feature templates ready to use

# Their CHUNKING benchmarks used the following parameters
# L2 regularization (C=1, rpho = sqrt(0.5))
# Stopping criterion for LBFGS: delta=1e-5, period=10
# Max iter for averaged perceptron = 50

# WHAT IS A TEMPLATE in CRFSuite
# Each element in the templates is a tuple/list of (name, offset) pairs,
# in which name presents a field name, and offset presents an offset to
# the current position.
# (('w', -2), ('w', 0)) extracts the value of 'w' field at two tokens
# to the left of the current position and then concatenates that to the
# value of the 'w' field at the current position (a broken trigram)
# And these are literally used as keys to a dictionary.
# The function feature_extractor receives a sequence of items (X in
# this example) read from the input data, and generates necessary
# attributes. The argument X presents a list of items; each item is
# represented by a mapping (dictionary) object from field names to
# their values. In the CoNLL chunking task, X[0]['w'] presents the
# word of the first item in the sequence, X[0]['pos'] presents the
# part-of-speech tag of the last item in the sequence.
# The mapping object of each item in X has a special key 'F' whose
# value is an attribute_list. Each element of an attribute_list must
# be a string or a tuple of (name, value) (an attribute with a
# weight). And the feature_templates are applied by using
# crfutils.apply_templates
# A basic example of this is given by
# python /home/prastog3/data/crfsuite/example/pos.py
train_tag_baseline_crf:
	echo TODO

#################################
####### WSJ TAG INPUT CREATOR
# The resolution was just to take the most probable one during training.
# LDC95T7
WSJ_PATH := /export/corpora/LDC/LDC99T42/treebank_3/tagged/pos/wsj
WSJ_POSTAG_CMD = python src/convert_wsj_files_to_single_file.py $(WSJ_PATH)
# I NEED TO FIX THE FOLLLOWING
# Currently I am assigning 46 pos tags in total
WSJTAG_CMD = $@ $@.tag.vocab $@.word.vocab
res/wsj_tag.train: 
	$(WSJ_POSTAG_CMD) 0 18  $(WSJTAG_CMD)
res/wsj_tag.dev:
	$(WSJ_POSTAG_CMD) 19 21 $(WSJTAG_CMD)
res/wsj_tag.test:
	$(WSJ_POSTAG_CMD) 22 24 $(WSJTAG_CMD)
test_wsj_tag_code:
	python src/convert_wsj_files_to_single_file.py 0 0 $$PWD/res/test 