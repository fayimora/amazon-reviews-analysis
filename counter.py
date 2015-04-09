from collections import Counter
from topic_modelling import get_tokeniser


if __name__ == '__main__':
    cnt = Counter()
    lines = open('data/topics_in.txt').readlines()
    tokeniser = get_tokeniser(ngram_range=(1,2))
    texts = map(lambda s: tokeniser(s), lines)

    print "Populating the counter"
    for text in texts:
        for token in text:
            cnt[token] += 1
