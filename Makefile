.PHONY: postagger_accuracy_small res/postag_small.model log/postag_small.pos.test.predict default_train
.SECONDARY:
PYCMD := OMP_NUM_THREADS=10 THEANO_FLAGS='floatX=float32,warn_float64=ignore,optimizer=fast_run,lib.amdlibm=True,mode=FAST_RUN,gcc.cxxflags=-O9 -L/home/prastog3/install/lib -I/home/prastog3/install/include,openmp=True' time python # -m pdb
NOSETEST_CMD := nosetests --verbosity=3 --with-doctest --exe --pdb --processes=1
test_all: # --failed --with-profile
	$(NOSETEST_CMD)  -w src/test
collect_test_all:
	$(NOSETEST_CMD) --collect-only -w src/test
# TARGET: tag stripped version of any input file in res folder

res/%.tagstrip: res/%
	sed 's#/[^ ]*##g' $< > $@
#####################################
### EVALUATION
# TARGET : Printed output of accuracy.
# SOURCE : 1. The predicted postags 2. The actual postags
# log/eval_tag_order0hmm~lbl10~LL~L1~0.001~0.2~0~NONE~toy.tag.vocab~toy.word.vocab~toy_sup~toy_unsup~toy.dev@toy.dev
# log/eval_tag_order0hmm~lbl10~LL~L1~0.001~0.2~0~NONE~wsj_tag.train.tag.vocab~wsj_tag.train.word.vocab~wsj_tag.train_sup~wsj_tag.train_unsup~wsj_tag.dev@wsj_tag.dev
# Evaluate on training done on truncated vocabulary.
# log/eval_tag_order0hmm~lbl10~LL~L1~0.001~0.2~0~NONE~wsj_tag.train.tag.vocab~wsj_tag.train.word.vocabtrunc~wsj_tag.train_sup~wsj_tag.train_unsup~wsj_tag.dev@wsj_tag.dev
log/eval_tag_% :
	$(MAKE) MYDEP1="log/predict_tag_$*.tagstrip" MYDEP2="res/$(call PREDICT_OPT_EXTRACTOR,2)" TARGET=$@ eval_tag_generic
eval_tag_generic: $(MYDEP1) $(MYDEP2)
	$(PYCMD) \
	src/eval_tag.py \
	    $(MYDEP1) \
	    $(MYDEP2) \
	    $(TARGET)

####################################
## POSTAGGER PREDICTION 
# TARGET : The output of the postagger 
# SOURCE : The trained postagger model,
#	   The raw words that we want to postag using this model
# log/predict_tag_order0hmm~lbl10~LL~L1~0.001~0.2~0~NONE~toy.tag.vocab~toy.word.vocab~toy_sup~toy_unsup~toy.dev@toy.dev.tagstrip
PREDICT_OPT_EXTRACTOR = $(word $1,$(subst @, ,$*))
log/predict_tag_% : 
	$(MAKE) MYDEP1="res/train_tag_$(call PREDICT_OPT_EXTRACTOR,1)" MYDEP2="res/$(call PREDICT_OPT_EXTRACTOR,2)" TARGET=$@ STOPAT=100 predict_tag_generic

predict_tag_generic: $(MYDEP1) $(MYDEP2)
	$(PYCMD) \
	src/predict_tag.py \
	    $(MYDEP1) \
	    $(MYDEP2) \
	    $(TARGET) \
	    $(STOPAT)
# # Train_train > Predict_train > Evaluate_train > Tune.parameters + Profile/debug.code (Loop)  > Train_train
####################################################################
# TAGGER TRAINING
# TARGET: The postagger model file.
# SOURCE: The sources are the vocabulary files and the tagged training files
# sample sources are the
# MODEL_TYPE = rhmm rmemm rsim # (Top down, Bottom up, Simultaneous)
# CPD_TYPE   = lbl # model of long context, addlambda<float>, unsmooth
# OBJECTIVE_TYPE      = LL, NCE # Log likelihood or NCE objective 
# PARAM_REG_TYPE      = L2, L1  # L2 or L1 regularization over the priors
# PARAM_REG_WEIGHT    = <float> # The weight of the regularization term
# UNSUP_LL_WEIGHT     = <float> # This is Multi conditional learning
# INIT_FROM_FILE      = {0,1}   # Init param from file or not
# PARAM_FILENAME      = name of yaml file that contains the parameters
# TAG_VOCAB_FILE      = name of the tag vocabulary file.
# WORD_VOCAB_FILE     = name of the word vocab file
# SUP_TRAIN_FILE      = name of the training file with supervised data
# UNSUP_TRAIN_FILE    = name of the training file with unsupervised data
# SUP_DEV_FILE        = name of the dev data file
# 2 example targets are
# res/train_tag_rhmm~addlambda0.1~LL~sgd~0.5~l1~0.01~wsj_tag.train.tag.vocab~wsj_tag.train.word.vocab~wsj_tag.train~unsup.txt~wsj_tag.dev~unsup_embedding.txt
# res/train_tag_order0hmm~lbl10~LL~L1~0.001~0.2~0~NONE~toy.tag.vocab~toy.word.vocab~toy_sup~toy_unsup~toy.dev
# SAVE_FILE            = name of the output pickle of the trained file
TRAIN_OPT_EXTRACTOR = $(word $1,$(subst ~, ,$*))
TRAIN_OPT_EXTRACTOR2 = $(word $1,$(subst ~, ,%))
default_train: res/train_tag_order0hmm~lbl100~LL~L1~0.001~0.2~0~NONE~wsj_tag.train.tag.vocab~wsj_tag.train.word.vocabtrunc~wsj_tag.train_sup~wsj_tag.train_unsup~wsj_tag.dev
res/train_tag_% : #res/wsj_tag.train.word.vocabtrunc res/wsj_tag.train.word.vocab
	$(PYCMD) src/train_tag.py MODEL_TYPE=$(call TRAIN_OPT_EXTRACTOR,1) \
	        CPD_TYPE=$(call TRAIN_OPT_EXTRACTOR,2) \
	        OBJECTIVE_TYPE=$(call TRAIN_OPT_EXTRACTOR,3) \
	        PARAM_REG_TYPE=$(call TRAIN_OPT_EXTRACTOR,4) \
	        PARAM_REG_WEIGHT=$(call TRAIN_OPT_EXTRACTOR,5) \
	        UNSUP_LL_WEIGHT=$(call TRAIN_OPT_EXTRACTOR,6) \
		INIT_FROM_FILE=$(call TRAIN_OPT_EXTRACTOR,7) \
		PARAM_FILENAME=res/$(call TRAIN_OPT_EXTRACTOR,8) \
	        TAG_VOCAB_FILE=res/$(call TRAIN_OPT_EXTRACTOR,9) \
	        WORD_VOCAB_FILE=res/$(call TRAIN_OPT_EXTRACTOR,10) \
	        SUP_TRAIN_FILE=res/$(call TRAIN_OPT_EXTRACTOR,11) \
	        UNSUP_TRAIN_FILE=res/$(call TRAIN_OPT_EXTRACTOR,12) \
	        SUP_DEV_FILE=res/$(call TRAIN_OPT_EXTRACTOR,13) \
	        SAVE_FILE=$@ 

#################################
## WSJ TAG INPUT CREATOR
# I just to take the most probable one during training, LDC95T7
# Note that I am also preprocessing things to take care of OOVs while
# converting to a single file 
WSJ_PATH := /export/corpora/LDC/LDC99T42/treebank_3/tagged/pos/wsj
WSJ_POSTAG_CMD = python src/datamunge/convert_wsj_files_to_single_file.py $(WSJ_PATH)
WSJTAG_CMD = $@ $@.tag.vocab $@.word.vocab
res/wsj_tag.train.word.vocabtrunc: res/wsj_tag.train.word.vocab
	head -n 10000 $< > $@ && echo "<OOV> 1" >> $@
res/wsj_tag.train_sup: res/wsj_tag.train
	cp $< $@
res/wsj_tag.train_unsup: res/wsj_tag.train
	sed 's#/[^ ]*##g' $< > $@
res/wsj_tag.train: 
	$(WSJ_POSTAG_CMD) 0 18  $(WSJTAG_CMD)
res/wsj_tag.dev:
	$(WSJ_POSTAG_CMD) 19 21 $(WSJTAG_CMD)
res/wsj_tag.test:
	$(WSJ_POSTAG_CMD) 22 24 $(WSJTAG_CMD)

#####################################
## BASELINE (CRFSuite)
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

# Lok at the bst feature set of the pos tagging paper with crfs
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
