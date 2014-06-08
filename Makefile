.PHONY: postagger_accuracy_small res/postag_small.model log/postag_small.pos.test.predict
.SECONDARY: 
PYCMD := python
RP := res/postag
MODELTYPE?= HNMM
OBJECTIVE_TYPE?= LL
OPTIMIZATION_TYPE?= LBFGS
UNSUP_LL_FACTOR?= 0.9
REGULARIZATION_FACTOR?= 0.1
REGULARIZATION_TYPE?= l2

# Perform training and testing using the small test data on 3 different types of models.
all:
	make -n postag_accuracy_small.GMEMM MODELTYPE=GMEMM && \
	make -n postag_accuracy_small.HNMM MODELTYPE=HNMM && \
	make -n postag_accuracy_small.SIMUL MODELTYPE=SIMUL

# SOURCE : 1. The predicted postags 2. The actual postags
# TARGET : Printed output of accuracy.
postag_accuracy_%.$(MODELTYPE) : log/postag_%.pos.test.predict.$(MODELTYPE) res/postag_%.pos.test
	$(PYCMD) src/postag_accuracy.py $+

# SOURCE : The trained postagger model,
#	   The raw words that we want to postag using this model
# TARGET : The output of the postagger 
log/postag_%.pos.test.predict.$(MODELTYPE) : $(RP)_%.model.$(MODELTYPE) $(RP)_%.word.test
	$(PYCMD) src/test_postag.py $(MODELTYPE) $+ $@

# TARGET: The postagger model file
# SOURCE: First two are vocabulary files. vocabulary of pos tagger and words
#         Next we have the supervised data
#	  Then we have the unsupervised data
res/postag_%.model.$(MODELTYPE) : $(RP)_%.word.vocab $(RP)_%.pos.vocab $(RP)_%.word.train $(RP)_%.pos.train $(RP)_%.raw_word
	$(PYCMD) src/train_postag.py $(MODELTYPE) $(OBJECTIVE_TYPE) $(OPTIMIZATION_TYPE) $+ $@ $(UNSUP_LL_FACTOR) $(REGULARIZATION_FACTOR) $(REGULARIZATION_TYPE)

