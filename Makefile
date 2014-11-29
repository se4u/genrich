.PHONY: postagger_accuracy_small res/postag_small.model log/postag_small.pos.test.predict default_train
.SECONDARY:
PYTEMPLATE = OMP_NUM_THREADS=10 THEANO_FLAGS='floatX=float32,warn_float64=ignore,optimizer=$1,lib.amdlibm=True,mode=$2,gcc.cxxflags=-$3 -L/home/prastog3/install/lib -I/home/prastog3/install/include,openmp=True,profile=$4' PYTHONPATH=$$PYTHONPATH:~/projects/genrich/src time python
PYCMD := $(call PYTEMPLATE,fast_run,FAST_RUN,O9,False)
PYLIGHT := $(call PYTEMPLATE,fast_compile,FAST_COMPILE,O1,False)
PYPROFILE := $(call PYTEMPLATE,fast_run,FAST_RUN,O9,True)
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
# This forces tuning of learning rate and batch size since the learning rate and the batch size are both 0. That triggers the batch size and learning rate tuning algorithm.
D_TUNE_LEARNING_ORDER0HMM := order0hmm~lbl10~LL~L2~0.001~0.2~0~NONE~wsj_tag.train.tag.vocab~wsj_tag.train.word.vocabtrunc~wsj_tag.train_sup~wsj_tag.train_unsup~wsj_tag.minivalidate~1234~0~10~0~sgd~25000~NOACTION
# The results of the order 0 model are the following.
# This is on validation data with 1/4 of DEV.
# Accuracy   : 87.393, Weighted Accuracy   : 84.169, Total: 4426
# IV Accuracy: 88.743, Weighted IV Accuracy: 84.586, Total: 4042
# OV Accuracy: 73.177, Weighted OV Accuracy: 73.177, Total: 384
# This has the best learning rate and batch size. for the order0hmm. This rate was found after using D_TUNE_LEARNING_RATE
D_BEST_LEARNING_ORDER0HMM := order0hmm~lbl10~LL~L2~0.001~0.2~0~NONE~wsj_tag.train.tag.vocab~wsj_tag.train.word.vocabtrunc~wsj_tag.train_sup~wsj_tag.train_unsup~wsj_tag.minivalidate~1234~1~10~0.04~sgd~100~NOACTION
D_TUNE_LEARNING_HNMM := order4rhmm~lbl10~LL~L2~0.001~0.2~0~NONE~wsj_tag.train.tag.vocab~wsj_tag.train.word.vocabtrunc~wsj_tag.train_sup~wsj_tag.train_unsup~wsj_tag.minivalidate~1234~1~10~0.04~sgd~100~NOACTION
DEFAULT := $(D_TUNE_LEARNING_HNMM)
log_eval: log/eval_tag_$(DEFAULT)@wsj_tag.from_500.validate
log/eval_tag_% :
	$(MAKE) MYDEP1="log/predict_tag_$*.tagstrip" MYDEP2="res/$(call PREDICT_OPT_EXTRACTOR,2)" TARGET=$@ eval_tag_generic
eval_tag_generic: $(MYDEP1) $(MYDEP2)
	$(PYCMD) \
	  src/eval_tag.py \
	    $(MYDEP1) \
	    $(MYDEP2) > $(TARGET) ; \
	tail -n 4 $(TARGET)

####################################
## POSTAGGER PREDICTION 
# TARGET : The output of the postagger 
# SOURCE : The trained postagger model,
#	   The raw words that we want to postag using this model
# Remove Stopat if you dont want it to stop
log_predict: log/predict_tag_$(DEFAULT)@wsj_tag.validate.tagstrip
PREDICT_OPT_EXTRACTOR = $(word $1,$(subst @, ,$*))
log/predict_tag_% : 
	$(MAKE) MYDEP1="res/train_tag_$(call PREDICT_OPT_EXTRACTOR,1)" MYDEP2="res/$(call PREDICT_OPT_EXTRACTOR,2)" TARGET=$@ STOPAT= predict_tag_generic

predict_tag_generic: $(MYDEP1) $(MYDEP2)
	$(PYCMD) \
	src/predict_tag.py \
	    $(MYDEP1) \
	    $(MYDEP2) \
	    $(TARGET) $(STOPAT)
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
# VALIDATION_FILE     = name of the dev data file
# RNG_SEED            = <int> Seed for the RNG
# BATCH_SIZE          = <int> The size of the minibatch
# EPOCH               = <int> Number of times to go over the training data
# LEARNING_RATE       = <float> rate update method, Initial learning rate.
# OPTIMIZATION_METHOD = ADAGRAD, SGD
# VALIDATION_FREQ     = <int> check performance on the validation data
#                             after this many batches
# VALIDATION_ACTION   = <string> Specify what we'd do after each validation step
# SAVE_FILE           = name of the output pickle of the trained file
TRAIN_OPT_EXTRACTOR = $(word $1,$(subst ~, ,$*))
TRAIN_OPT_EXTRACTOR2 = $(word $1,$(subst ~, ,%))
TRAIN_TAG_CMD = MODEL_TYPE=$(call TRAIN_OPT_EXTRACTOR,1) \
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
	        VALIDATION_FILE=res/$(call TRAIN_OPT_EXTRACTOR,13) \
		RNG_SEED=$(call TRAIN_OPT_EXTRACTOR,14) \
		BATCH_SIZE=$(call TRAIN_OPT_EXTRACTOR,15) \
		EPOCH=$(call TRAIN_OPT_EXTRACTOR,16) \
		LEARNING_RATE=$(call TRAIN_OPT_EXTRACTOR,17) \
		OPTIMIZATION_METHOD=$(call TRAIN_OPT_EXTRACTOR,18) \
		VALIDATION_FREQ=$(call TRAIN_OPT_EXTRACTOR,19) \
		VALIDATION_ACTION=$(call TRAIN_OPT_EXTRACTOR,20)

TRAIN_CMD = src/train_tag.py \
		$(TRAIN_TAG_CMD) \
	        SAVE_FILE=$@
## TRAINING SHORTCUTS
# TARGET: By calling any one of these I can either do quick compile or
# turn on profiling of the code. 
res_train: res/train_tag_$(DEFAULT)
quick_train: quick/train_tag_$(DEFAULT)
profile_train: profile/train_tag_$(DEFAULT)
# SOURCE: res/wsj_tag.train.word.vocabtrunc res/wsj_tag.train.word.vocab
res/train_tag_% : 
	$(PYCMD) \
		$(TRAIN_CMD)
quick/train_tag_% : 
	$(PYLIGHT) \
		$(TRAIN_CMD)
profile/train_tag_% : 
	$(PYPROFILE) \
		$(TRAIN_CMD)
## DECODE PARAMETER STRING
# TARGET: The encode and decode are perfect reverses of each other.
# 	The decoder convert a string to a verbose template file
#	The encoder converts the template file to a string
# USAGE: make -s encode_train.setting
#        make -s decode_(result of encode)
decode_%:
	for param in $(TRAIN_TAG_CMD); do echo $$param; done
## ENCODE PARAMETER STRING
# TARGET: The % is a template file that contains all the parameters in
# a verbose form. That file is converted to a single string on the
# basis of the logic in TRAIN_TAG_CMD by actually parsing the
# TRAIN_TAG_CMD string itself. This is done to avoid duplication of
# code. 
encode_%:
	python src/datamunge/encode_train_template_to_string.py $*
#################################
## WSJ TAG INPUT CREATOR
# I just to take the most probable one during training, LDC95T7
# Note that I am also preprocessing things to take care of OOVs while
# converting to a single file
# There are a milion token in the training file, and 36K types.
# Out of that the vocabtrunc file constrains to only the top 10K types.
WSJ_PATH := /export/corpora/LDC/LDC99T42/treebank_3/tagged/pos/wsj
WSJ_POSTAG_CMD = python src/datamunge/convert_wsj_files_to_single_file.py $(WSJ_PATH)
WSJTAG_CMD = $@ $@.tag.vocab $@.word.vocab
wsj_data: res/wsj_tag.train.word.vocabtrunc res/wsj_tag.train_sup res/wsj_tag.train_unsup res/wsj_tag.validate
res/wsj_tag.train.word.vocabtrunc: res/wsj_tag.train.word.vocab
	head -n 10000 $< > $@ && echo "<OOV> 1" >> $@
res/wsj_tag.train.word.vocab: res/wsj_tag.train
res/wsj_tag.train_sup: res/wsj_tag.train 
	cp $< $@
res/wsj_tag.train_unsup: res/wsj_tag.train
	sed 's#/[^ ]*##g' $< > $@
res/wsj_tag.train: src/datamunge/convert_wsj_files_to_single_file.py
	$(WSJ_POSTAG_CMD) 0 18  $(WSJTAG_CMD)
res/wsj_tag.from_%.validate: res/wsj_tag.dev
	awk '{if(NR>$* && NR < 500+$*){print $$0}}' $< > $@
res/wsj_tag.validate: res/wsj_tag.dev
	head -n 500 $< > $@
res/wsj_tag.dev: src/datamunge/convert_wsj_files_to_single_file.py
	$(WSJ_POSTAG_CMD) 19 21 $(WSJTAG_CMD)
res/wsj_tag.test:
	$(WSJ_POSTAG_CMD) 22 24 $(WSJTAG_CMD)
# wc res/suffix_count_3  res/suffix_count_2 res/suffix_count_4 res/prefix_count_3  res/prefix_count_2 res/prefix_count_4  
#   1422   2844  17064 res/suffix_count_3
#    333    666   3663 res/suffix_count_2
#   3199   6398  41587 res/suffix_count_4
#   1682   3364  20184 res/prefix_count_3
#    312    624   3432 res/prefix_count_2
#   3593   7186  46709 res/prefix_count_4
#  10541  21082 132639 total
res/prefix_count_%: res/wsj_tag.train.tagstrip
	awk '{for(i=1; i<=NF; i++){if(length($$i) > $*){print substr($$i, 0, $*)}}}' $< | egrep  '^[a-zA-Z]*$$' | sort | uniq -c | sort -nr | awk '{if($$1 > 5){print $$0}}' > $@
res/suffix_count_%: res/wsj_tag.train.tagstrip
	awk '{for(i=1; i<=NF; i++){if(length($$i) > $*){print substr($$i, length($$i)-$*+1)}}}' $< | egrep  '^[a-zA-Z]*$$' | sort | uniq -c | sort -nr | awk '{if($$1 > 5){print $$0}}'  > $@
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
