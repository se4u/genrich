.PHONY: postagger_accuracy_small res/postag_small.model log/postag_small.pos.test.predict
.SECONDARY:
PYCMD := python
# According to Apps Hungarian
# mo = model(object)
# it = iterator
# cl = class
# Perform training and testing using the small test data on 3 different types of models.
# all: # Train_train > Predict_train > Evaluate_train > Tune.parameters + Profile/debug.code (Loop)  > Train_train > Predict_dev

#####################################
### EVALUATION
# TARGET : Printed output of accuracy.
# SOURCE : 1. The predicted postags 2. The actual postags
# Write your own code. It's simple and even though it may differ from
# the final eval it would still be intuitive while development.
postag_accuracy_%.$(MODELTYPE) : log/postag_%.pos.test.predict.$(MODELTYPE) res/postag_%.pos.test
	$(PYCMD) src/postag_accuracy.py $+

####################################
## POSTAGGER PREDICTION 
# TARGET : The output of the postagger 
# SOURCE : The trained postagger model,
#	   The raw words that we want to postag using this model
log/postag_%.pos.test.predict.$(MODELTYPE) : $(RP)_%.model.$(MODELTYPE) $(RP)_%.word.test
	$(PYCMD) src/test_postag.py $(MODELTYPE) $+ $@

####################################################################
# Pylearn2 represents the while loop itself as a class with three
# things.
# The algorithm, the model, save_path, save_freq, "extensions".
# I think these are the most important.
# The training algorithms are,
# What is interesting to learn is how the problem has been parametized
# and the classes defined quite beautifully. The algorithms have a
# base class, (which specifies only three function continue_training,
# setup, train) and then the default. Then there are learning_rules
# for sgd, The main sgd algorithm is
# pylearn2.training_algorithms.sgd.SGD
# but there is also batch gradient descent.

# The objective is written as a Theano function. The only problem is
# that here I would have a faily large number of parameters. So a
# simple problem could be optimizing a max-ent lm using theano. If I
# can do that using theano. (and pylearn2) then I can do this also.
# For this part I found the following code which I can learn from
# https://github.com/ddahlmeier/neural_lm/blob/master/lbl.py
# https://github.com/ddahlmeier/neural_lm/blob/master/lbl_nce.py
# https://github.com/turian/neural-language-model
# https://github.com/gwtaylor/theano-rnn

# My model has a objective. and during training since I want to
# leverage both supervised and unsupervised data therefore I need to
# write 3 kinds of scores
# 1. score_sto: sentence, tag observed but some of the tags can be
# missing. this is the most general case.
# 2. score_ao: 
# 3. score_so: 
# (It calculates this using the observed data, and parameters)

# Theano helps me calculate gradients automatically.
# As long as I can write the score (calculated using DP or not) as a
# theano function. It is possible by using scan in theano.
# I need two types of gradients. The gradient I calculate given a
# supervised data point and the gradient with unsuperrvised datapoint.

# My model has a method to give gradient (wrt parameter)
# (Given observed data only, and parameters) # Calculate the gradients using Theano

# Train the model using pylearn2/scipy.optimize
# pylearn2 allows me to quickly switch between different optimization
# methods. along with tricks of the trade that are a part of
# optimization business.

# Perform prediction using whatever framework.
# My model can predict hidden var. (Given observed var, parameters)

# The prediction method would be used in calculating the objective/score
# anyway. This would be necessary because the objective would be an expectation.
# So the parameters are an intrinsic part of the model.

# Probability of tag given the previous sequence of words.
# And we are assuming that the parametrization is the appropriate
# parametrization to use. The important thing is that we can
# "efficiently" predict. so we need to maintain that. Infact code for
# that.

# So generate toy data. and call make on that target all the time.
# Train the model and then use it as a LM, (calculate the probability
# efficiently) and get the perpelexity, also make an eval method that
# gives a tagged sequence, and write code to evaluate it. 
####################################
# TAGGER TRAINING
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
# MODEL_TYPE = rhmm rmemm rsim # (Top down, Bottom up, Simultaneous) Generative PD
# CPD_TYPE   = lbl, addlambda<float>, unsmooth, chengoodman # Smoothing methods for the long context CPD
# OBJECTIVE_TYPE      = LL, NCE # Log likelihood or NCE objective 
# OPTIMIZATION        = em, sgd, naturalgrad #LBFGS, EM, SGD, Natural Gradient
# UNSUP_LL_WEIGHT     = <float> # This is Multi conditional learning
# PARAM_REG_TYPE      = L2, L1  # L2 or L1 regularization over the priors
# PARAM_REG_WEIGHT    = <float> # The weight of the regularization term
# TAG_VOCAB_FILE      = name of the tag vocabulary file.
# WORD_VOCAB_FILE     = name of the word vocab file
# SUP_TRAIN_FILE      = name of the training file with supervised data
# UNSUP_TRAIN_FILE    = name of the training file with unsupervised data
# SUP_DEV_FILE        = name of the dev data file
# WORD_EMBEDDING_FILE = name of file that contains word embeddings
# 2 example targets are
# res/train_tag_rhmm~addlambda0.1~LL~sgd~0.5~l1~0.01~wsj_tag.train.tag.vocab~wsj_tag.train.word.vocab~wsj_tag.train~unsup.txt~wsj_tag.dev~unsup_embedding.txt
# res/train_tag_rhmm~addlambda0.1~LL~sgd~0.5~l1~0.01~toy.tag.vocab~toy.word.vocab~toy_sup~toy_unsup~toy.dev~toy_embedding
# SAVE_FILE            = name of the output pickle of the trained file
# So most of the non-convex functions I've optimized have been done
# with LBFGS optimization with one crucial change: any time the
# optimizer thinks its converged, dump the history cache and force it
# to flush the current approximation of the inverse hessian and take
# just a normal gradient step. Most of the Berkeley NLP papers since
# 2006 which do LBFGS non-convex optimization have used this trick and
# found it pretty important I believe. 
TRAIN_OPT_EXTRACTOR = $(word $1,$(subst ~, ,$*))
res/train_tag_% :
	export TAG_VOCAB_FILE=res/$(call TRAIN_OPT_EXTRACTOR,8) \
	       WORD_VOCAB_FILE=res/$(call TRAIN_OPT_EXTRACTOR,9) \
	       SUP_TRAIN_FILE=res/$(call TRAIN_OPT_EXTRACTOR,10) \
	       UNSUP_TRAIN_FILE=res/$(call TRAIN_OPT_EXTRACTOR,11) \
	       SUP_DEV_FILE=res/$(call TRAIN_OPT_EXTRACTOR,12) \
	       WORD_EMBEDDING_FILE=res/$(call TRAIN_OPT_EXTRACTOR,13) && \
	$(MAKE) MODEL_TYPE=$(call TRAIN_OPT_EXTRACTOR,1) \
	        CPD_TYPE=$(call TRAIN_OPT_EXTRACTOR,2) \
	        OBJECTIVE_TYPE=$(call TRAIN_OPT_EXTRACTOR,3) \
	        OPTIMIZATION_TYPE=$(call TRAIN_OPT_EXTRACTOR,4) \
	        UNSUP_LL_WEIGHT=$(call TRAIN_OPT_EXTRACTOR,5) \
	        PARAM_REG_TYPE=$(call TRAIN_OPT_EXTRACTOR,6) \
	        PARAM_REG_WEIGHT=$(call TRAIN_OPT_EXTRACTOR,7) \
	        TAG_VOCAB_FILE=$$TAG_VOCAB_FILE \
	        WORD_VOCAB_FILE=$$WORD_VOCAB_FILE \
	        SUP_TRAIN_FILE=$$SUP_TRAIN_FILE \
	        UNSUP_TRAIN_FILE=$$UNSUP_TRAIN_FILE \
	        SUP_DEV_FILE=$$SUP_DEV_FILE \
	        WORD_EMBEDDING_FILE=$$WORD_EMBEDDING_FILE \
	        SAVE_FILE=$@ \
	        MYDEP="$$VOCAB_FILE $$TRAIN_FILE $$SUP_TRAIN_FILE $$UNSUP_TRAIN_FILE $$WORD_EMBEDDING_FILE" \
	train_tag_generic

train_tag_generic: $(MYDEP)
	$(PYCMD) src/train_tag.py MODEL_TYPE=$(MODEL_TYPE) \
	CPD_TYPE=$(CPD_TYPE) OBJECTIVE_TYPE=$(OBJECTIVE_TYPE) \
	OPTIMIZATION_TYPE=$(OPTIMIZATION_TYPE) \
	UNSUP_LL_WEIGHT=$(UNSUP_LL_WEIGHT) \
	PARAM_REG_TYPE=$(PARAM_REG_TYPE) \
	PARAM_REG_WEIGHT=$(PARAM_REG_WEIGHT) \
	TAG_VOCAB_FILE=$(TAG_VOCAB_FILE) \
	WORD_VOCAB_FILE=$(WORD_VOCAB_FILE) \
	SUP_TRAIN_FILE=$(SUP_TRAIN_FILE) \
	UNSUP_TRAIN_FILE=$(UNSUP_TRAIN_FILE) \
	SUP_DEV_FILE=$(SUP_DEV_FILE) \
	WORD_EMBEDDING_FILE=$(WORD_EMBEDDING_FILE) \
	SAVE_FILE=$(SAVE_FILE) 

# Jason said that how are you sure that the feature set they are using
# is good ? and which features are useful for hmms ?
# I had promised certain thigns are made.
# Accuracy pipeline
# Basic api, a ll, dp based optimizer.
# Let's say I dont get Theano working ?
# I said that I could
# So do enough that I can say that I have a HMM with the features
# working. The experimentation would have to come as well. But just
# implement the rhmm for now. 
#####################################
# Lok at the bst feature set of the pos tagging paper with crfs
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
WSJ_POSTAG_CMD = python src/datamunge/convert_wsj_files_to_single_file.py $(WSJ_PATH)
# I NEED TO FIX THE FOLLLOWING
# Currently I am assigning 46 pos tags in total
WSJTAG_CMD = $@ $@.tag.vocab $@.word.vocab
res/wsj_tag.train: 
	$(WSJ_POSTAG_CMD) 0 18  $(WSJTAG_CMD)
res/wsj_tag.dev:
	$(WSJ_POSTAG_CMD) 19 21 $(WSJTAG_CMD)
res/wsj_tag.test:
	$(WSJ_POSTAG_CMD) 22 24 $(WSJTAG_CMD)
# test_wsj_tag_code:
# 	python src/datamunge/convert_wsj_files_to_single_file.py 0 0 $$PWD/res/test 