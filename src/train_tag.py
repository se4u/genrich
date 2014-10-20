import sys, itertools
from util_oneliner import get_vocab_from_file
import test_tag
# fn means filename
options=dict(e.split("=") for e in sys.argv[1:])
model_type        = options["MODEL_TYPE"]
cpd_type          = options["CPD_TYPE"]
objective_type    = options["OBJECTIVE_TYPE"]
optimization_type = options["OPTIMIZATION_TYPE"]
unsup_ll_weight   = float(options["UNSUP_LL_WEIGHT"])
param_reg_type    = options["PARAM_REG_TYPE"]
param_reg_weight  = float(options["PARAM_REG_WEIGHT"])
tag_vocab         = get_vocab_from_file(open( \
                        options["TAG_VOCAB_FILE"], "rb"))
word_vocab        = get_vocab_from_file(open( \
                        options["WORD_VOCAB_FILE"], "rb"))
sup_train_fn      = options["SUP_TRAIN_FILE"]
unsup_train_fn    = options["UNSUP_TRAIN_FILE"]
sup_dev_fn        = options["SUP_DEV_FILE"]
word_embedding_fn = options["WORD_EMBEDDING_FILE"]
save_fn           = options["SAVE_FILE"]
eta               = 0.01
num_pass          = 5

def model_factory(model_type):
    if model_type=="order0hmm":
        from model.tag_order0hmm import tag_order0hmm
        return tag_order0hmm
    elif model_type=="rhmm":
        raise NotImplementedError
        from model.tag_rhmm import tag_rhmm
        return tag_rhmm
    else:
        raise NotImplementedError
        
def get_sentence_tag_from_row(row):
    sentence=[word.split(r"/")[0] for word in row]
    tag=[word.split(r"/")[1] for word in row]
    return sentence, tag

def eval_on_train_dev(msg, mo_model, sup_train_file, sup_dev_file):
    print >> sys.stderr, msg
    test_tag.evaluate_model(mo_model, sup_train_file)
    test_tag.evaluate_model(mo_model, sup_dev_file)
    return

def train_on_file(mo_model, train_fn, num_pass, eta, dev_fn, filetype):
    for pass_ in xrange(num_pass):
        for row in open(train_fn, "rb"):
            row=row.strip().split()
            if filetype=="sup":
                sentence, tag = get_sentence_tag_from_row(row)
                mo_model.update_parameter(mo_model.gradient_sto(sentence, tag)*eta)
            elif filetype=="unsup":
                mo_model.update_parameter(mo_model.gradient_so(row)*eta)
            else:
                raise ValueError(filetype)
        eval_on_train_dev("train_on_file, Pass: %d"%pass_, mo_model, train_fn, dev_fn)
    return

def train_on_twofile(mo_model, sup_train_fn, unsup_train_fn, num_pass, eta, sup_dev_fn):
    for pass_ in xrange(num_pass):
        for i, row in enumerate(itertools.izip_longest(open(sup_train_fn, "rb"), open(unsup_train_fn, "rb"))):
            if row is None:
                continue
            elif i%2==0:
                row=row.strip().split()
                sentence, tag = get_sentence_tag_from_row(row)
                mo_model.update_parameter(mo_model.gradient_sto(sentence, tag)*eta)
            elif i%2==1:
                row=row.strip().split()
                mo_model.update_parameter(mo_model.gradient_so(row)*eta)
        eval_on_train_dev("train_on_twofile, Pass: %d"%pass_, mo_model, sup_train_fn, sup_dev_fn)
    return

# It is a switch that only needs the types of things.
cl_model = model_factory(model_type)
mo_model=cl_model(cpd_type, objective_type, param_reg_type,
                  param_reg_weight,
                  unsup_ll_weight,
                  initialize_embedding_according_to_file=True,
                  initialize_param_to_random=True,
                  word_embedding_filename=word_embedding_fn,
                  tag_vocab=tag_vocab,
                  word_vocab=word_vocab)

train_on_file(mo_model, sup_train_fn, num_pass, eta, sup_dev_fn, "sup")
train_on_file(mo_model, unsup_train_fn, num_pass, eta, sup_dev_fn, "unsup")

mo_model.reinitialize()
train_on_file(mo_model, unsup_train_fn, num_pass, eta, sup_dev_fn, "unsup")
train_on_file(mo_model, sup_train_fn, num_pass, eta, sup_dev_fn, "sup")

mo_model.reinitialize()
train_on_twofile(mo_model, sup_train_fn, unsup_train_fn, num_pass, eta, sup_dev_fn)
