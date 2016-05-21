from SheffUtils import JsonReader
from nltk.translate import bleu_score
from spacy.en import English
from collections import Counter
################################## word movers distance
import os
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from scipy.spatial.distance import cosine
from sklearn.metrics import euclidean_distances
from pyemd import emd

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
            f.write(w.encode('utf-8'))
            f.write('\n'.encode('utf-8'))
            #print(w, file=f)
            pass
    del wv

W = np.memmap("data/embed.dat", dtype=np.double, mode="r", shape=(3000000, 300))
with open("data/embed.vocab") as f:
    vocab_list = map(str.strip, f.readlines())

vocab_dict = {w: k for k, w in enumerate(vocab_list)}

def wmd(s1, s2):
    vect = CountVectorizer(stop_words="english").fit([s1, s2])
    
    v_1, v_2 = vect.transform([s1, s2])
    v_1 = v_1.toarray().ravel()
    v_2 = v_2.toarray().ravel()

    W_ = W[[vocab_dict[w] if w in vocab_dict else vocab_dict[vocab_dict.keys()[0]] for w in vect.get_feature_names()]]

    D_ = euclidean_distances(W_)

    # pyemd needs double precision input
    v_1 = v_1.astype(np.double)
    v_2 = v_2.astype(np.double)
    D_ = D_.astype(np.double)

    return emd(v_1, v_2, D_)
################################## WMD

fname = 'snli_1.0_test'

reader = JsonReader('./snli_1.0/'+fname+'.jsonl')
nlp = English()

noLabel = 0.
correctLabel = 0.
wrongLabel = 0.

def bleuScore(s1, s2):
    return bleu_score.sentence_bleu(s1, s2)

def similarityScore(s1, s2):
    return s1.similarity(s2)

def lenghtRatio(s1,s2):
	return float(min(len(s1),len(s2)))/float(max(len(s1),len(s2)))

# the ratio of pair of words across the premise and hypothesis which share a POS tag, as a real value
# number of pairs of words:
def crossUnigramsRatio(s1, s2):
    nPairs = min(len(s1),len(s2))
    l2 = [w2.pos_ for w2 in s2]
    cnt = 0.
    for w in s1:
        if w.pos_ in l2:
            cnt += 1.
            idx = l2.index(w.pos_)
            l2.pop(idx)
    cuRatio = cnt / nPairs
    return cuRatio

with open('myOutput_'+fname+'.csv','w') as outFile:
    outFile.write('label,bleu,similarity,wmd,crossUnigrams\n')
    for snt in reader:
        if snt['gold_label'] != '-':
            s1 = nlp(snt['sentence1'])
            s2 = nlp(snt['sentence2'])

            a = bleu_score.sentence_bleu(s1[:].text, s2[:].text)
            b = s1.similarity(s2)
            c = wmd(s1.text, s2.text)
            d = crossUnigramsRatio(s1, s2)

            outFile.write('%s,%f,%f,%f,%f\n'%(snt['gold_label'],a,b,c,d))

        else:
            noLabel += 1.

print('Done calculating values')
