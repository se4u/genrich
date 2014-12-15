"""Given a saved model file and an untagged dev/test file make prediction of the tags.
"""
# Created: 31 October 2014
__author__  = "Pushpendre Rastogi"
__contact__ = "pushpendre@jhu.edu"
import sys, itertools, signal, time, logging, re, yaml
from util_oneliner import tictoc
global starting_time
global sentence_done

starting_time=None
sentence_done=0
model_to_load=sys.argv[1]
untagged_filename=sys.argv[2]
save_filename=sys.argv[3]
logging.basicConfig(filename=save_filename+".logger", level=logging.DEBUG)
logger=logging
try:
    sentence_limit=int(sys.argv[4])
except:
    logger.debug("Setting sentence limit to unlimited\n")
    sentence_limit=None
def signal_handler(signal, frame):
    logger.debug("\nReceived C-c, Aborting\nTime: %0.1f, Sentence: %d\n"%(
            time.time()-starting_time, sentence_done))
    sys.exit(1)
    return

signal.signal(signal.SIGINT, signal_handler)

# mo_ means model object
# tm means tag model
with tictoc("Init tag_order0hmm"):
    if re.search("order([0-9]+)rhmm", model_to_load):
        #from model.tag_rhmm import tag_rhmm
        from model.tag_rhmm import tag_rhmm
        with tictoc("Loading model yaml"):
            init_data=yaml.load(open(model_to_load, "rb"))
        mo_tm=tag_rhmm(test_time=True, init_data=init_data)
    elif re.search("order([0-9]+)hmm", model_to_load):
        from model.tag_order0hmm import tag_order0hmm
        mo_tm=tag_order0hmm(None, None, None, None, None,
                            True, model_to_load,
                            None, None)
    else:
        raise NotImplementedError(model_to_load)

# Open the output as a line buffered file
total_tokens=0
with tictoc("Tagging sentences"):
    with open(save_filename, "wb", 1) as outf:
        starting_time=time.time()
        for i,sentence in enumerate(open(untagged_filename)):
            if sentence_limit is not None and i >= sentence_limit:
                break
            words=sentence.strip().split()
            if len(words)<2:
                continue
            total_tokens+=len(words)
            with tictoc("Predicting tags"):
                if words[-1]==".":
                    words=words[:-1]
                tags,score=mo_tm.predict_posterior_tag_sequence(words)
            isoov=[False if e in mo_tm.word_vocab else True
                   for e in words]
            outf.write(" ".join(r"/".join([w,t,str(s),str(o)])
                                for w,t,s,o
                                in itertools.izip(words, tags, score,isoov)))
            outf.write("\n")
            sentence_done=i
            sys.stderr.write(".")
    logger.debug("Predicted %d sentences with %d tokens"%(sentence_done,
                  total_tokens))
