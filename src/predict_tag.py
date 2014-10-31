"""Given a saved model file and an untagged dev/test file make prediction of the tags.
"""
# Created: 31 October 2014
__author__  = "Pushpendre Rastogi"
__contact__ = "pushpendre@jhu.edu"

import sys
model_to_load=sys.argv[1]
untagged_filename=sys.argv[2]
save_filename=sys.argv[3]

