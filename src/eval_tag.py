""" This evaluates the predicted tags against the gold standard and
reports following scores
1. Accuracy
2. Weighted Accuracy (scores assigned to correct prediction by total score)
"""
import sys, itertools
predicted_tag_file=open(sys.argv[1], "rb")
actual_tag_file=open(sys.argv[2], "rb")
"""a means actual
   p means predicted
   w means word
   t means tag
   s means score
   we_ means weighted"""
correct=0.0
we_correct=0.0
total=0.0
we_total=0.0 
for actual_row, predicted_row in itertools.izip(actual_tag_file,
                                                predicted_tag_file):
    for a, p in itertools.izip(actual_row.split(), predicted_row.split()):
        a_w, a_t=a.split(r"/")
        p_w, p_t, p_s=p.split(r"/")
        assert a_w==p_w, "\n".join([a,p,a_w,p_w])
        accuracy=1.0 if a_t==p_t else 0.0
        correct+=accuracy
        we_correct+=accuracy*float(p_s)
        total+=1
        we_total+=float(p_s)
        sys.stdout.write(r"/".join([a_w, a_t, p_t, p_s, str(accuracy)])+" ")
    sys.stdout.write("\n")
sys.stderr.write(
    "Accuracy: %0.3f, Weighted Accuracy: %0.3f\n"%(100*correct/total,
                                                 100*we_correct/we_total))
