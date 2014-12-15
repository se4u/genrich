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
   o means OOV or not
   we_ means weighted
   ov means out of vocabulary
   iv means in vocabulary
"""
correct_iv=0.0
we_correct_iv=0.0
total_iv=0.0
we_total_iv=0.0

correct_ov=0.0
we_correct_ov=0.0
total_ov=0.0
we_total_ov=0.0
Template = """
Accuracy   : %0.3f, Weighted Accuracy   : %0.3f, Total: %0.0f
IV Accuracy: %0.3f, Weighted IV Accuracy: %0.3f, Total: %0.0f
OV Accuracy: %0.3f, Weighted OV Accuracy: %0.3f, Total: %0.0f
"""
for actual_row, predicted_row in itertools.izip(actual_tag_file,
                                                predicted_tag_file):
    for a, p in itertools.izip(actual_row.split(), predicted_row.split()):
        a_w, a_t=a.split(r"/")
        p_w, p_t, p_s, p_o=p.split(r"/")
        assert a_w==p_w, "\n".join([a,p,a_w,p_w])
        accuracy=1.0 if a_t==p_t else 0.0
        iv = False if p_o == "True" else True

        if iv:
            correct_iv+=accuracy
            we_correct_iv+=accuracy*float(p_s)
            total_iv+=1
            we_total_iv+=float(p_s)
        else:
            correct_ov+=accuracy
            we_correct_ov+=accuracy*float(p_s)
            total_ov+=1
            we_total_ov+=float(p_s)
        sys.stdout.write(r"/".join([a_w, a_t, p_t, p_s, p_o, str(accuracy)])+" ")
    sys.stdout.write("\n")

sys.stdout.write(
    Template%(100*(correct_iv+correct_ov)/(total_iv+total_ov),
              100*(we_correct_iv+we_correct_ov)/(we_total_iv+we_total_ov),
              (total_iv+total_ov),
              100*(correct_iv)/(total_iv),
              100*(we_correct_iv)/(we_total_iv),
              total_iv,
              -1 if total_ov ==0 else 100*(correct_ov)/(total_ov),
              -1 if we_total_ov == 0 else 100*(we_correct_ov)/(we_total_ov),
              total_ov))
