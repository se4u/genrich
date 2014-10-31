import sys, itertools
from util_oneliner import get_vocab_from_file
import test_tag
# fn means filename
options=dict(e.split("=") for e in sys.argv[1:])

def train_model(mo_model):
    warn("Actually implement this")
    return mo_model

if __name__=="__main__":
    model_type = options["MODEL_TYPE"]
    if model_type=="order0hmm":
        from model.tag_order0hmm import tag_order0hmm
        cl_model = tag_order0hmm
    elif model_type=="rhmm":
        raise NotImplementedError
        from model.tag_rhmm import tag_rhmm
        cl_model = tag_rhmm
    else:
        raise NotImplementedError
    
    mo_model=cl_model(cpd_type=options["CPD_TYPE"],
                      objective_type=options["OBJECTIVE_TYPE"],
                      param_reg_type=options["PARAM_REG_TYPE"],
                      param_reg_weight=float(options["PARAM_REG_WEIGHT"]),
                      unsup_ll_weight=float(options["UNSUP_LL_WEIGHT"]),
                      initialize_param_according_to_file=int(
            options["INIT_FROM_FILE"]),
                      param_filename=options["PARAM_FILENAME"],
                      tag_vocab=get_vocab_from_file(open(
                options["TAG_VOCAB_FILE"], "rb")),
                      word_vocab=get_vocab_from_file(open(
                options["WORD_VOCAB_FILE"], "rb")))
    train_model(mo_model,
                options["SUP_TRAIN_FILE"],
                options["UNSUP_TRAIN_FILE"],
                options["SUP_DEV_FILE"])
    mo_model.save(options["SAVE_FILE"])
