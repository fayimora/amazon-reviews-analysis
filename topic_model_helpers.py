import glob
import re
from gensim.models.ldamodel import LdaModel
from topic_modelling import load


class TopicModelHelpers:
    def __init__(self, fnames, model=None, corpus=None, dictionary=None):
        """`fnames` is an array of files for [lda_model, distribution]"""
        self.reviews = open('data/electronics_topics_in.txt').readlines()

        print "Loding topic model..."
        if model is not None:
            print "Using argument model"
            self.lda = model
        else:
            self.lda = LdaModel.load(fnames[0])

        if corpus is not None:
            print "Using argument corpus and dictionary"
            self.corpus = corpus
            self.dictionary = dictionary
        else:
            print "Loading corpus and dictionary from file"
            self.corpus = load("data/models/electronics_tfidf_corpus.pkl")
            self.dictionary = load("data/models/electronics_dict.pkl")

        print "Loading review-topic distribution..."
        self.review_dist = [l for l in self.lda[self.corpus]]
        tmp = lambda dist: sorted(dist, key=lambda arr: arr[1], reverse=True)
        self.review_dist = map(lambda dist: tmp(dist), self.review_dist)

        print "processing topics"
        tmp = map(lambda t: re.sub("(\d*\.\d*\*)", "", t), self.lda.show_topics(-1))
        self.topics = map(lambda ts: re.sub("\\s\+", ",", ts), tmp)
        # self.topics.reverse()

    def get_reviews_in_topic(self, topic_id, threshold=0.20):
        """Return reviews with at least `threshold` proportion of `topic_id`"""
        review_ids = []
        for i, topic_dist in enumerate(self.review_dist):
            for topic, per in topic_dist:
                if topic == topic_id and per >= threshold:
                    review_ids.append((i, per))
                    break

        res = map(lambda (i, per): (per, self.reviews[i]), review_ids)
        return res

    def filter_reviews(self, token, topic_id, threshold):
        """ This function takes a token and a topic_id. It returns the reviews
        that have a proportion of the argument topic and contains the argument token."""
        return filter(lambda s: token in s[1].lower(), self.get_reviews_in_topic(topic_id, threshold=threshold))


