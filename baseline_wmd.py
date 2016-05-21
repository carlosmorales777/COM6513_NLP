import logging
import os

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-4s %(message)s', filename='finalEntailment.log')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logging.getLogger('').addHandler(ch)

if not os.path.exists("data/embed.dat"):
    print("Caching word embeddings in memmapped format...")
    #from gensim import models
    from gensim.models.word2vec import Word2Vec
    wv = Word2Vec.load_word2vec_format("data/GoogleNews-vectors-negative300.bin",
        binary=True)
    wv.init_sims(replace=True) # recommended new step?
    fp = np.memmap("data/embed.dat", dtype=np.double, mode='w+', shape=wv.syn0norm.shape)
    fp[:] = wv.syn0norm[:]
    with open("data/embed.vocab", "w") as f:
        for _, w in sorted((voc.index, word) for word, voc in wv.vocab.items()):
            #f.write(w.encode('utf-8'))
            #f.write('\n'.encode('utf-8'))
            print(w, file=f)
    del wv

W = np.memmap("data/embed.dat", dtype=np.double, mode="r", shape=(3000000, 300))
with open("data/embed.vocab") as f:
    vocab_list = map(str.strip, f.readlines())

vocab_dict = {w: k for k, w in enumerate(vocab_list)}

d1 = "Obama speaks to the media in Illinois"
d2 = "The President addresses the press in Chicago"

vect = CountVectorizer(stop_words="english").fit([d1, d2])
print("Features:",  ", ".join(vect.get_feature_names()))

from scipy.spatial.distance import cosine
v_1, v_2 = vect.transform([d1, d2])
v_1 = v_1.toarray().ravel()
v_2 = v_2.toarray().ravel()
print(v_1, v_2)
print("cosine(doc_1, doc_2) = {:.2f}".format(cosine(v_1, v_2)))

from sklearn.metrics import euclidean_distances
W_ = W[[vocab_dict[w] for w in vect.get_feature_names()]]
D_ = euclidean_distances(W_)
print("d(addresses, speaks) = {:.2f}".format(D_[0, 7]))
print("d(addresses, chicago) = {:.2f}".format(D_[0, 1]))

from pyemd import emd

# pyemd needs double precision input
v_1 = v_1.astype(np.double)
v_2 = v_2.astype(np.double)
v_1 /= v_1.sum()
v_2 /= v_2.sum()
D_ = D_.astype(np.double)
D_ /= D_.max()  # just for comparison purposes
print("d(doc_1, doc_2) = {:.2f}".format(emd(v_1, v_2, D_)))

