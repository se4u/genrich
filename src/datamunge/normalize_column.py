"""Given a space delimited file, it divides the first column which is numeric by the sum of first column, therefore converting counts to probabilities, rest of the line is printed as is
"""
__date__    = "11 December 2014"
__author__  = "Pushpendre Rastogi"
__contact__ = "pushpendre@jhu.edu"

import sys
data=[e.strip().split()+[""] for e in sys.stdin]
total=sum([float(e[0]) for e in data])
for e in data:
    sys.stdout.write("%10.4f %s\n"%(float(e[0])/total, " ".join(e[1:])))
