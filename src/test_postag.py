import sys
model_type=sys.argv[1]
model_file=open(sys.argv[2], "rb")
test_word_file=open(sys.argv[3], "rb")
predicted_tag_file=open(sys.argv[4], "wb")
