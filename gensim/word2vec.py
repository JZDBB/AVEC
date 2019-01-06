import gzip
import gensim
import logging
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.externals import joblib
# logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)

def read_input(input_file):
    """This method reads the input file which is in gzip format"""

    logging.info("reading file {0}...this may take a while".format(input_file))
    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate(f):

            if (i % 10000 == 0):
                logging.info("read {0} reviews".format(i))
            # do some pre-processing and return list of words for each review
            # text
            yield [gensim.utils.simple_preprocess(line)]


def read_data(root):
    dirs = os.listdir(root)
    w = []
    for d in dirs:
        curr = os.path.join(root, d)
        name = d.split('_')[0]
        if name == '402':
            a = 1
        csv_file = name + '_TRANSCRIPT.csv'
        with open(os.path.join(curr, csv_file), 'r') as f:
            f.readline()
            n = 0
            while True:
                if n == 35:
                    a = 1
                l = f.readline().lower().replace('\n', '').replace('\r', '').replace('((', '(').replace('))', ')').split('\t')[-1]
                if '(' not in l:
                    l = l.split(' ')
                else:
                    l = l.split('(')[1].split(')')[0].split(' ')
                if l == ['']:
                    break
                w.append(l)
                n += 1
    return w


data_dir = '../data/10001'
documents = read_data(data_dir)
for d in documents:
    if 'listens' in d:
        print('in')

model = gensim.models.Word2Vec(documents,
                                size=300,
                                window=5,
                                min_count=1,
                                workers=10)
model.train(documents, total_examples=len(documents), epochs=1000)
word_dict = {}
for w in model.wv.vocab:
    word_dict[w] = model[w]
joblib.dump(word_dict,'word.pkl',compress = 3)
print(model.wv.most_similar(positive='mom'))
visualizeWords = ['a', 'what', 'an', 'when', 'dad', 'mom', 'mother', 'one', 'two', 'thanks', 'nephew', 'i', 'me', 'you',
                  'were', 'was', 'are', 'is', '<synch>', '<laughter>', 'today', 'to', 'talk', 'here', 'learn', 'with', 'just', 'mixture']
visualizeVecs = [model.wv[word] for word in visualizeWords]
temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
covariance = 1.0 / len(visualizeVecs) * temp.T.dot(temp)
U, S, V = np.linalg.svd(covariance)
coord = temp.dot(U[:, 0:2])

for i in range(len(visualizeWords)):
    plt.text(coord[i, 0], coord[i, 1], visualizeWords[i], bbox=dict(facecolor='green', alpha=0.1))

plt.xlim((np.min(coord[:, 0]), np.max(coord[:, 0])))
plt.ylim((np.min(coord[:, 1]), np.max(coord[:, 1])))
plt.show()