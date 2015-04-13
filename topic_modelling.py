import logging
import re
import cPickle
from os.path import isfile
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
    lda = LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word(dictionary), \
                    alpha='auto', update_every=1, chunksize=10000, passes=1, iterations=50)
    return lda


def train_lsi_model(corpus, dictionary):
    lsi = LsiModel(corpus=corpus, id2word=id2word(dictionary), num_topics=20, chunksize=10000,\
                   onepass=True)
    return lsi


def get_tokeniser(ngram_range=(1,1), stop_words='english'):
    vect = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words)
    return vect.build_analyzer()


def prettify(topics):
    return map(lambda ts: re.sub("\\s\+", ",", ts), map(lambda t: re.sub("(\d*\.\d*\*)", "", t), topics))


def save(data, file):
    fo = open(file, 'w')
    cPickle.dump(data, fo, protocol=2)
    fo.close()


def load(file):
    fo = open(file, 'rb')
    data = cPickle.load(fo)
    fo.close()
    return data


if __name__ == '__main__':
    dataset_name = "electronics"
    tfidf_corpus_path = 'data/models/%s_tfidf_corpus.pkl' % dataset_name
    tfidf_corpus = None

    if not isfile(tfidf_corpus_path):
        logging.warn("No saved models available. Generating corpus. This might take a while...")

        logging.info("Loading data from text file")
        lines = open('data/%s_topics_in.txt' % dataset_name).readlines()

        logging.info("Tokenising data")
        tokeniser = get_tokeniser()
        texts = map(lambda s: tokeniser(s), lines)

        # logging.info("Saving corpus texts")
        # save(texts, 'data/models/%s_texts.pkl' % dataset_name)

        logging.info("Generating bag of words")
        dictionary = Dictionary(texts)
        save(dictionary, 'data/models/%s_dict.pkl' % dataset_name)
        corpus = [dictionary.doc2bow(text) for text in texts]

        # logging.info("Saving bag of words")
        # save(corpus, 'data/models/%s_corpus.pkl' % dataset_name)

        logging.info("Applying Tfidf model to corpus")
        tfidf = TfidfModel(corpus)
        tfidf_corpus = [tfidf[c] for c in corpus]

        logging.info("Saving TF-IDF transformed corpus")
        save(tfidf_corpus, tfidf_corpus_path)
    else:
        logging.info("Loading saved TF-IDF dictionary")
        dictionary = load('data/models/%s_dict.pkl' % dataset_name)
        logging.info("Loading saved TF-IDF corpus")
        tfidf_corpus = load(tfidf_corpus_path)

    # logging.info("Training LSI model")
    # lsi = train_lsi_model(tfidf_corpus, dictionary)
    # lsi_topics = prettify(lsi.show_topics(-1))
    # lsi_topic_distribution = [l for l in lsi[tfidf_corpus]]

    logging.info("Training LDA model")
    lda = train_lda_model(tfidf_corpus, dictionary, 30)
    lda_raw_topics = lda.show_topics(-1)
    lda_pretty_topics = prettify(lda.show_topics(-1))
    lda_topic_distribution = [l for l in lda[tfidf_corpus]]


