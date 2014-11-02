"""Given a saved model file and an untagged dev/test file make prediction of the tags.
"""
# Created: 31 October 2014
__author__  = "Pushpendre Rastogi"
__contact__ = "pushpendre@jhu.edu"
import sys, os, itertools, signal,time
from util_oneliner import tictoc
starting_time=None
sentence_done=0
def signal_handler(signal, frame):
    sys.stderr.write("\nTime: %0.1f, Sentence: %d\n"%(
            time.time()-starting_time, sentence_done))
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)
sys.path.append(os.path.join(os.getcwd(), "src", "model"))
from tag_order0hmm import tag_order0hmm
model_to_load=sys.argv[1]
untagged_filename=sys.argv[2]
save_filename=sys.argv[3]
try:
    sentence_limit=int(sys.argv[4])
except:
    sentence_limit=None
# mo_ means model object
# tm means tag model
with tictoc("Init tag_order0hmm"):
    mo_tm=tag_order0hmm(None, None, None, None, None,
                        True, model_to_load,
                        None, None)

# Open the output as a line buffered file
total_tokens=0
with tictoc("Tagging %d Sent with %d tokens"%(sentence_done,total_tokens)):
    with open(save_filename, "wb", 1) as outf:
        starting_time=time.time()
        for i,sentence in enumerate(open(untagged_filename)):
            if sentence_limit is not None and i >= sentence_limit:
                break
            words=sentence.strip().split()
            total_tokens+=len(words)
            tags,score=mo_tm.predict_posterior_tag_sequence(words)
            outf.write(" ".join(r"/".join([w,t,str(s)])
                                for w,t,s
                                in itertools.izip(words, tags, score)))
            outf.write("\n")
            sentence_done=i
            sys.stderr.write(".")
