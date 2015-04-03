import logging
from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora import Dictionary
from gensim.models.lsimodel import LsiModel


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def train_lsi_model(corpus, dictionary):
    id2word = {}
    for word in dictionary.token2id:
        id2word[dictionary.token2id[word]] = word

    lsi = LsiModel(corpus=corpus, id2word=id2word)
    return lsi


def get_tokeniser(ngram_range=(1,1), stop_words='english'):
    vect = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words)
    return vect.build_tokenizer()


if __name__ == '__main__':
    lines = open('data/lda_in_noid.txt').readlines()
    tokeniser = get_tokeniser()
    texts = map(lambda s: tokeniser(s), lines)
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lsa = train_lsi_model(corpus, dictionary)



