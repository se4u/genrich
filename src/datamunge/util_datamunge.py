from util_oneliner import strlist_to_int



def word_vocab_and_embedding(word_vocab_file):
    word=[e.split(" ")[0] for e in word_vocab_file]
    assert word[-1]==r"</s>"
    word=word[:-1]
    word_vocab_file.seek(0)
    embedding=strlist_to_int(e.strip().split(" ")[1:]
                             for e
                             in word_vocab_file)
    return [word, embedding[:-1], embedding[-1]]
