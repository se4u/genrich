import sys
from glob import glob
from os import path
from collections import defaultdict
wsj_base_dir=sys.argv[1]
start_dir=int(sys.argv[2])
end_dir=int(sys.argv[3])
wordtag_file=open(sys.argv[4], "wb")
tagvocab_file=open(sys.argv[5], "wb")
wordvocab_file=open(sys.argv[6], "wb")
pos_dict=defaultdict(int)
word_dict=defaultdict(int)
POS_TAG_PRIORITY = "NN > IN > NNP > DT > JJ > NNS > CD > RB > VBD > VB > CC > TO > VBZ > VBN > PRP > VBP > MD > POS > PRP$ > WDT > JJR > NNPS > WP > WRB > JJS > RBR > RP".split(" > ")
def pos_tag_priority_tolerant_index(e):
    try:
        return POS_TAG_PRIORITY.index(e)
    except ValueError:
        return len(POS_TAG_PRIORITY)

def pick_high_priority_tag(taglist):
    tag_rank=[(e, pos_tag_priority_tolerant_index(e)) for e in taglist]
    return min(tag_rank, key=lambda x: x[1])[0]

def add_to_dictionary(row):
    for e in row:
        if e != "":
            try:
                [e1, e2]=e.split(r"/")
            except:
                raise Exception(e)
            word_dict[e1]+=1
            pos_dict[e2]+=1
    return

def has_digits(w, num):
    return sum(e.isdigit() for e in w) > num

def mark_certain_words_numeric(w):
    if has_digits(w, 2):
            w="<NUM>"
    return w

def handle_piped_tags(row):
    """The input is an actual string with tagged words
    """
    ret_arr=[]
    for e in row.split():
        [a, b]=e.split(r"/")
        if "|" in b:
            b=pick_high_priority_tag(b.split("|"))
        a=mark_certain_words_numeric(a.lower())
        ret_arr.append(a+"/"+b)
    return ret_arr

def get_sets(fh):
    start=True
    ret=[]
    for row in fh:
        row=row.strip()
        for pt in [("[ ",""),
                   ("Chiat\/NNP", "Chiat/NNP"),
                   (r"\/", "^"),
                   (r"S*/NNP&P/NN", r"S&P/NNP"),
                   (r"AT*/NNP&T/NN", r"AT&T/NNP"),
                   (" ]","")]:
            row=row.replace(pt[0], pt[1])
        if row=="":
            continue
        if row=="======================================":
            if start:
                start=False
                continue
            else:
                yield " ".join(ret)
                ret=[]
        else:
            # First handle the piped tags
            ret_arr=handle_piped_tags(row)
            ret.append(" ".join(ret_arr))
            add_to_dictionary(ret_arr)
    yield " ".join(ret)

for directory in xrange(start_dir, end_dir+1):
    dir_name="%02.d"%directory
    for f in glob(path.expanduser(path.join(wsj_base_dir, dir_name, '*.pos'))):
        for sentence in get_sets(open(f)):
            print >> wordtag_file, sentence

add_to_dictionary([r"<OOV>/<OOV>"])
for k,v in sorted(pos_dict.items(), key=lambda(x): x[1], reverse=True):
    print >> tagvocab_file, k, v
print ""
for k, v in sorted(word_dict.items(), key=lambda(x): x[1], reverse=True):
    print >> wordvocab_file, k, v
    
