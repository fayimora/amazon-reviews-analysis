import logging
import re
from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora import Dictionary
from gensim.models.lsimodel import LsiModel
from gensim.models.ldamodel import LdaModel
from gensim.models.tfidfmodel import TfidfModel


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def id2word(dictionary):
    id2word = {}
    for word in dictionary.token2id:
        id2word[dictionary.token2id[word]] = word
    return id2word


def train_lda_model(corpus, dictionary):
    lda = LdaModel(corpus=corpus, id2word=id2word(dictionary), num_topics=30)
    return lda


def train_lsi_model(corpus, dictionary):
    lsi = LsiModel(corpus=corpus, id2word=id2word(dictionary), num_topics=30)
    return lsi


def get_tokeniser(ngram_range=(1,1), stop_words='english'):
    vect = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words)
    return vect.build_analyzer()


def prettify(topics):
    return map(lambda ts: re.sub("\\s\+", ",", ts), map(lambda t: re.sub("(\d*\.\d*\*)", "", t), topics))


if __name__ == '__main__':
    logging.info("Loading data from text file")
    lines = open('data/lda_in_noid.txt').readlines()

    logging.info("Tokenising data")
    tokeniser = get_tokeniser()
    texts = map(lambda s: tokeniser(s), lines)

    logging.info("Generating bag of words")
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    logging.info("Applying Tfidf model to corpus")
    tfidf = TfidfModel(corpus)
    tfidf_corpus = [tfidf[c] for c in corpus]

    logging.info("Training LSI model")
    lsi = train_lsi_model(corpus, dictionary)
    lsi_topics = prettify(lsi.show_topics())

    logging.info("Training LSA model")
    lda = train_lda_model(corpus, dictionary)
    lda_topics = prettify(lda.show_topics())


