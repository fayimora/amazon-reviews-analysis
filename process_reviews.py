import gzip
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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


def topics(raw_data, out_folder, name, id=False):
    fname = '%s_topics_in_id.txt' % name if id else '%s_topics_in.txt' % name
    fout = open(out_folder+fname, 'wa')

    logging.info('Parsing raw data and processing it')

    for review in parse(raw_data):
        if review:
            if id:
                line = review['review/userId'] + " == " + review['review/text'] + "\n"
            else:
                line = review['review/text'] + "\n"
            fout.write(line)
    fout.close()


if __name__ == '__main__':
    topics('data/Electronics.txt.gz', 'data/', 'electronics')


