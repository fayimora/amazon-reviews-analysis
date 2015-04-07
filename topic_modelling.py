import logging
import re
from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora import Dictionary
from gensim.models.lsimodel import LsiModel
from gensim.models.ldamodel import LdaModel
from gensim.models.tfidfmodel import TfidfModel


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', \
                    level=logging.INFO)


def id2word(dictionary):
    id2word = {}
    for word in dictionary.token2id:
        id2word[dictionary.token2id[word]] = word
    return id2word


def train_lda_model(corpus, dictionary, num_topics):
    lda = LdaModel(corpus=corpus, id2word=id2word(dictionary), num_topics=num_topics, \
                   alpha='auto', chunksize=10000, passes=5, iterations=1000)
    return lda


def train_lsi_model(corpus, dictionary):
    lsi = LsiModel(corpus=corpus, id2word=id2word(dictionary), num_topics=10)
    return lsi


def get_tokeniser(ngram_range=(1,2), stop_words='english'):
    vect = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words)
    return vect.build_analyzer()


def prettify(topics):
    return map(lambda ts: re.sub("\\s\+", ",", ts), map(lambda t: re.sub("(\d*\.\d*\*)", "", t), topics))


if __name__ == '__main__':
    logging.info("Loading data from text file")
    lines = open('data/topics_in.txt').readlines()

    logging.info("Tokenising data")
    tokeniser = get_tokeniser()
    texts = map(lambda s: tokeniser(s), lines)

    logging.info("Generating bag of words")
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    logging.info("Applying Tfidf model to corpus")
    tfidf = TfidfModel(corpus)
    tfidf_corpus = [tfidf[c] for c in corpus]

    # logging.info("Training LSI model")
    # lsi = train_lsi_model(tfidf_corpus, dictionary)
    # lsi_topics = prettify(lsi.show_topics(-1))
    # lsi_topic_distribution = [l for l in lsi[tfidf_corpus]]

    logging.info("Training LDA model")
    lda = train_lda_model(tfidf_corpus, dictionary, 30)
    lda_topics = prettify(lda.show_topics(-1))
    lda_topic_distribution = [l for l in lda[tfidf_corpus]]


