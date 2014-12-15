"""This saves the trained tagger model to a mat file for introspection
"""
__date__    = "10 December 2014"
__author__  = "Pushpendre Rastogi"
__contact__ = "pushpendre@jhu.edu"

""" TODO: Make this code more general
"""
import tag_rhmm, yaml, sys
model_yaml_file=sys.argv[1]
out_file=sys.argv[2]
yaml_data=yaml.load(open(model_yaml_file, 'rb'))
tag_rhmm.save_to_mat_file(yaml_data, out_file)
