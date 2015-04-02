import gzip
import simplejson

def parse(filename):
    f = gzip.open(filename, 'r')
    entry = {}
    for l in f:
        l = l.strip()
        colonPos = l.find(':')
        if colonPos == -1:
            yield entry
            entry = {}
            continue
        eName = l[:colonPos]
        rest = l[colonPos+2:]
        entry[eName] = rest
    yield entry


reviews = []
lda_in = open('data/lda_in.txt', 'wa')
for e in parse("data/Jewelry.txt.gz"):
    review_json = simplejson.loads(simplejson.dumps(e))
    reviews.append(review_json)
    line = review_json['review/userId'] + " == " + review_json['review/text'] + "\n"
    # line = review_json['review/text'] + "\n"
    lda_in.write(line)

lda_in.close()
