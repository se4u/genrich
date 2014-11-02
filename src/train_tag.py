import sys, numpy.linalg, logging, numpy, signal, time
from util_oneliner import get_vocab_from_file, tictoc, mean
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

starting_time=None
sentence_done=None
def signal_handler(signal, frame):
    sys.stderr.write("\nTime: %0.1f, Sentence: %d\n"%(
            time.time()-starting_time, sentence_done))
    sys.exit(1)
signal.signal(signal.SIGINT, signal_handler)
# fn means filename
options=dict(e.split("=") for e in sys.argv[1:])

def train_model(mo_model, sup_train_fn, unsup_train_fn, sup_dev_fn, eta):
    """
    """
    global sentence_done
    for sentence_done, row in enumerate(open(sup_train_fn, "rb")):
        row=row.strip().split()
        if len(row)==0:
            continue
        words=[e.split("/")[0] for e in row]
        tags=[e.split("/")[1] for e in row]
        with tictoc("Training update time"):
            delta=mo_model.update_ao(eta=eta, tags=tags, words=words)
        logging.debug(" ".join([str(numpy.linalg.norm(e)) for e in delta]))
        
        with tictoc("Prediction time"):
            prediction=mo_model.predict_posterior_tag_sequence(words)
            predicted_tags=prediction[0]
            predicted_scores=prediction[1]
        logging.debug(str(mean(map(lambda x,y: 1 if x==y else 0,
                                   predicted_tags, tags))))
        logging.debug("\n".join(["/".join([a,b,c,str(d)])
                                 for a,b,c,d
                                 in zip(words, tags,
                                        predicted_tags,predicted_scores)]))
                      
        if any(numpy.isnan(e).any() for e in delta):
            raise Exception("delta became nan!")
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
    
    with tictoc("Init mo_model"):
        mo_model=cl_model(cpd_type=options["CPD_TYPE"],
                      objective_type=options["OBJECTIVE_TYPE"],
                      param_reg_type=options["PARAM_REG_TYPE"],
                      param_reg_weight=float(options["PARAM_REG_WEIGHT"]),
                      unsup_ll_weight=float(options["UNSUP_LL_WEIGHT"]),
                      init_from_file=int(options["INIT_FROM_FILE"]),
                      param_filename=options["PARAM_FILENAME"],
                      tag_vocab=\
                          get_vocab_from_file(open(options["TAG_VOCAB_FILE"],
                                                   "rb")),
                      word_vocab=\
                          get_vocab_from_file(open(options["WORD_VOCAB_FILE"],
                                                   "rb")),
                      test_time=False)
    
    with tictoc("Training mo_model"):
        train_model(mo_model,
                    options["SUP_TRAIN_FILE"],
                    options["UNSUP_TRAIN_FILE"],
                    options["SUP_DEV_FILE"],
                    0.001)
    
    with tictoc("Saving model"):
        mo_model.save(options["SAVE_FILE"])
