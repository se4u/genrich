"""This file encodes a template to a string that is used as the makefile target
"""
# Created: 06 November 2014
__author__  = "Pushpendre Rastogi"
__contact__ = "pushpendre@jhu.edu"
import sys
templace_fn=sys.argv[1]
makefile = open("Makefile", "rb").read().split("\n")
st_idx=[i for i,row in enumerate(makefile) if row.startswith("TRAIN_TAG_CMD = ")]
assert len(st_idx)==1
st_idx=st_idx[0]
end_idx=st_idx+min([i for i,row in enumerate(makefile[st_idx:])
         if not row.endswith("\\")])
makefile[st_idx]=makefile[st_idx][len("TRAIN_TAG_CMD = "):]
train_tag_cmd = [e.replace("\\","").strip() for e in makefile[st_idx:end_idx+1]]
d={}
for i,e in enumerate(train_tag_cmd):
    assert int(e.split(",")[1][:-1])==i+1
    d[e.split("=")[0]]=i
setting_list=[None]*len(d)
for setting in open(templace_fn, "rb"):
    key, value=setting.strip().split("=")
    if value.startswith("res/"):
        value=value[len("res/"):]
    setting_list[d[key]]=value
print "~".join(setting_list)
