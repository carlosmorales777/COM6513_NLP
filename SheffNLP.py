import logging
import numpy as np
#from numpy import random
import nltk
from nltk.corpus import stopwords
import string
from collections import Counter
import re
import gensim
from nltk.translate import bleu_score
from gensim.models.word2vec import Word2Vec

################################## word movers distance
import os
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from scipy.spatial.distance import cosine
from sklearn.metrics import euclidean_distances
from pyemd import emd

##################################### wmd


class NlpEngine:
    def __init__(self):
        # this RE performed ok during testing and requires less resources
		# than nltk.tokenize.word_tokenize(text)
		self.nonWords = re.compile("[^a-zA-Z]")
		self.STOPWORDS = set(nltk.corpus.stopwords.words('english')
							+ ["n't","'s"])
		self.SYMBOLS = set(" ".join(string.punctuation).split(" ") 
							+ ['\'\'','``','--','&nbsp','nbsp'])

	# we need to tokenize our text so...
    def tokenise(self, text):
		tokens = []
		ntext = self.nonWords.split(text)  # nltk.tokenize.word_tokenize(text)
		for token in ntext:
		    # lets focus in words larger than 2 chars
			if len(token.strip()) > 2:
				tokens.append(token.strip().lower())
        # stop words
		tokens = [token for token in tokens if token not in self.STOPWORDS]

        # stop symbols
		tokens = [token for token in tokens if token not in self.SYMBOLS]

		while "\\n" in tokens:
			tokens.remove('\\n')

		return tokens

class W2V:
	"""
	word2vec class.
	This model perfoms summing of word vectors when the word exists in our model
	"""
	def __init__(self):
		self.model = ''
		self.notInModel = Counter()
		self.countNot = 0
		self.countYes = 0
	
	def getWordSum(self, word):
		if word in self.model.vocab:
			self.countYes += 1
			return sum(self.model[word])
		elif word.capitalize() in self.model.vocab: # check capitalized version
			self.countYes += 1
			return sum(self.model[word.capitalize()])
		else:
			self.notInModel[word] += 1
			self.countNot += 1
			return False

	def wordsNotInModel(self):
		logging.info('Stats: %d in model and %d missing'
						%(self.countYes, self.countNot))
		with open('wNotInM.txt','w') as of:
			sortedWords = sorted(self.notInModel.items(),
								key=lambda ww:ww[1], reverse=True)
			for item in sortedWords:
				of.write('%s:%d\n'%(item[0],item[1]))
		logging.info('Dumped words to file.')

class DocumentStore:
    def __init__(self, tokenWeights = True, extraFeatures = True, EXTRA_WEIGHTS_LABELS = [
    'bleuScore', 'similarityScore', 'wordMoversDistance', 'crossUnigramsRatio']):
        self.words = {}
        self.words2 = {}  # hypothesis words
        self.wordId = 0
        self.wordId2 = 0  # hypothesis
        self.extraFeatures = {} # for our new features
        self.docId = 0
        self.documents = {}
        self.tokenWeights = tokenWeights
        self.extraFeatures = extraFeatures
        self.EXTRA_WEIGHTS_LABELS = EXTRA_WEIGHTS_LABELS
        #####################
        if not os.path.exists("data/embed.dat"):
            print("Caching word embeddings in memmapped format...")
            #from gensim import models
            from gensim.models.word2vec import Word2Vec
            wv = Word2Vec.load_word2vec_format("data/GoogleNews-vectors-negative300.bin.gz",
                binary=True)
            wv.init_sims(replace=True) # recommended new step?
            fp = np.memmap("data/embed.dat", dtype=np.double, mode='w+', shape=wv.syn0.shape)
            fp[:] = wv.syn0[:]
            with open("data/embed.vocab", "w") as f:
                for _, w in sorted((voc.index, word) for word, voc in wv.vocab.items()):
                    f.write(w.encode('utf-8'))
                    f.write('\n'.encode('utf-8'))
                    #print(w, file=f)
                    pass
            del wv

        self.W = np.memmap("data/embed.dat", dtype=np.double, mode="r", shape=(3000000, 300))
        with open("data/embed.vocab") as f:
            self.vocab_list = map(str.strip, f.readlines())

        self.vocab_dict = {w: k for k, w in enumerate(self.vocab_list)}
        #####################

    def addDocument(self, document):
        # our document needs 2 counters, one for each sentence
        doc = {'s1' : Counter(), 's2' : Counter(), 'extraFeatures' : {} }
		
        for word in document.getTokens():
			# if this word does not exists in our general collection
            if word not in self.words:
				# assign an Id to this word
                self.words[word] = self.wordId
				# increase the Id count
                self.wordId += 1
			# increase this word's weight for this doc
            doc['s1'][self.words[word]] += 1

        # we do the same for the hypothesis
        for word2 in document.getTokens2():
            if word2 not in self.words2:
                self.words2[word2] = self.wordId2
                self.wordId2 += 1
            doc['s2'][self.words2[word2]] += 1

        # compute extra features scores
        for label in self.EXTRA_WEIGHTS_LABELS:
            if label == 'bleuScore' or label == 'wordMoversDistance':
                v = getattr(self, label)(document.sentence1.text, document.sentence2.text)
            else:
                v = getattr(self, label)(document.sentence1, document.sentence2)
            doc['extraFeatures'][label] = v
            #logging.debug('feature label: %s value: %f'%(label, v))
        # store document
        self.documents[self.docId] = doc
        self.docId += 1
		# return the assigned Id for the document
        return self.docId - 1

    def generateMatrix(self):
        """
        Our weights matrix will now be formed by 3 'partitions'

        Each row represents a document
        the features are now formed by sentence 1 words + sentence 2 words + our new 4 extra scores.
        """
        if self.tokenWeights and self.extraFeatures:
            nFeatures = self.wordId + self.wordId2 + len(self.EXTRA_WEIGHTS_LABELS)
            logging.info('Exporting TOKEN WEIGHTS AND EXTRA FEATURES %dx%d'%(self.docId, nFeatures))
            mtrx = np.zeros((self.docId, nFeatures))
        
            for docId, doc in self.documents.iteritems():
                # iterate through 1st sentence
                for wId, val in doc['s1'].iteritems():
                    mtrx[docId, wId] = val
                # then iterate thru 2nd sentence, store on 2ND PARTITION
                for wId, val in doc['s2'].iteritems():
                    mtrx[docId, self.wordId + wId] = val
                # finally extra features values stored at the end of the vector
                for label, val in doc['extraFeatures'].iteritems():
                    mtrx[docId, self.wordId + self.wordId2 + self.EXTRA_WEIGHTS_LABELS.index(label)] = val

        elif self.tokenWeights and not self.extraFeatures:
            nFeatures = self.wordId + self.wordId2
            logging.info('Exporting TOKEN WEIGHTS %dx%d'%(self.docId, nFeatures))
            mtrx = np.zeros((self.docId, nFeatures))
        
            for docId, doc in self.documents.iteritems():
                # iterate through 1st sentence
                for wId, val in doc['s1'].iteritems():
                    mtrx[docId, wId] = val
                # then iterate thru 2nd sentence, store on 2ND PARTITION
                for wId, val in doc['s2'].iteritems():
                    mtrx[docId, self.wordId + wId] = val
        else:
            nFeatures = len(self.EXTRA_WEIGHTS_LABELS)
            logging.info('Exporting EXTRA FEATURES %dx%d'%(self.docId, nFeatures))
            mtrx = np.zeros((self.docId, nFeatures))
        
            for docId, doc in self.documents.iteritems():
                for label, val in doc['extraFeatures'].iteritems():
                    mtrx[docId, self.EXTRA_WEIGHTS_LABELS.index(label)] = val
        logging.info('Matrix generated')
        logging.info(mtrx.shape)
        return mtrx

    def lenghtRatio(self, s1,s2):
		return float(min(len(s1),len(s2)))/float(max(len(s1),len(s2)))

	# the ratio of pair of words across the premise and hypothesis which share a POS tag, as a real value
	# number of pairs of words:
    def crossUnigramsRatio(self, s1, s2):
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

    def bleuScore(self, s1, s2):
        return bleu_score.sentence_bleu(s1, s2)

    def similarityScore(self, s1, s2):
        return s1.similarity(s2)

    def wordMoversDistance(self, s1, s2):
        vect = CountVectorizer(stop_words="english").fit([s1, s2])
        
        v_1, v_2 = vect.transform([s1, s2])
        v_1 = v_1.toarray().ravel()
        v_2 = v_2.toarray().ravel()

        W_ = self.W[[self.vocab_dict[w] if w in self.vocab_dict else self.vocab_dict[self.vocab_dict.keys()[0]] for w in vect.get_feature_names()]]
        D_ = euclidean_distances(W_)

        v_1 = v_1.astype(np.double)
        v_2 = v_2.astype(np.double)
        D_ = D_.astype(np.double)
        
        return emd(v_1, v_2, D_)

    def totalWords(self):
        return self.wordId + self.wordId2

#    def printWords(self):#
#       print self.words
#        print self.words2

    def totalDocuments(self):
        return self.docId

    def vectorize(self, sentence1, sentence2):
        if self.tokenWeights and self.extraFeatures:
            vector = np.zeros(self.wordId + self.wordId2 + len(self.EXTRA_WEIGHTS_LABELS))
            # first we traverse sentence 1
            for word in sentence1:
                # we only assign a value if this word was seen on training
                if word in self.words:
                    vector[self.words[word]] += 1
            # then we go through sentence 2
            for word in sentence2:
                # we only assign a value if this word was seen on training
                if word in self.words2:
                    vector[self.wordId + self.words2[word]] += 1
            # then we add the extra values
            #for i in range(len(self.EXTRA_WEIGHTS_LABELS)):
            for label in self.EXTRA_WEIGHTS_LABELS:
                if label == 'bleuScore' or label == 'wordMoversDistance':
                    v = getattr(self, label)(sentence1.text, sentence2.text)
                else:
                    v = getattr(self, label)(sentence1, sentence2)
                vector[self.wordId + self.wordId2 + self.EXTRA_WEIGHTS_LABELS.index(label)] = v
        elif self.tokenWeights and not self.extraFeatures:
            vector = np.zeros(self.wordId + self.wordId2)
            # first we traverse sentence 1
            for word in sentence1:
                # we only assign a value if this word was seen on training
                if word in self.words:
                    vector[self.words[word]] += 1
            # then we go through sentence 2
            for word in sentence2:
                # we only assign a value if this word was seen on training
                if word in self.words2:
                    vector[self.wordId + self.words2[word]] += 1
        else:
            vector = np.zeros(len(self.EXTRA_WEIGHTS_LABELS))
            for label in self.EXTRA_WEIGHTS_LABELS:
                if label == 'bleuScore' or label == 'wordMoversDistance':
                    v = getattr(self, label)(sentence1.text, sentence2.text)
                else:
                    v = getattr(self, label)(sentence1, sentence2)
                vector[self.EXTRA_WEIGHTS_LABELS.index(label)] = v
        return vector
                                 
#    def topNwords(self, vector, n, reverse = False):
#        """
#        Sort by value and return top elements
#        """
#        logging.info(vector)
#        wordsWeights = []
#        for wrd,idx in self.words.iteritems():
#            # store index, word, weight
#            wordsWeights.append([wrd,vector[idx]])
#        # sort
#        topWs = sorted(wordsWeights,key=lambda ww:ww[1], reverse=reverse)[0:n]
#        # find top 10
#        return topWs

    def releaseTraingingData(self):
        """
        We force the release of these two elements to keep memory usage low
        """
        del(self.documents)
        #del(self.sumsOfVectors)
        self.documents = {}
        #self.sumsOfVectors = {}

# Aiming for functionallity decoupling...
# Perceptron does not need to know about the origin of our features. It must
# deal only with a vector of weights. It should not care for where these
# wheights come from or how they are computed
class PerceptronEntailment:
    def __init__(self, classLabels):
        self.weightsIdx = 1
        self.classLabels = classLabels

    def test(self, weights, testLabels, tests):
        # set the received weights
        self.weights = weights
        # initialize counts
        correct = 0
        wrong = 0
        for n in range(len(testLabels)):
            predicted = self.predictLabel(tests[n], bias = True)
            if testLabels[n] == predicted:
                correct += 1
            else:
                wrong += 1
        return correct,wrong

    def predictLabel(self, vector, numWeight = None, bias = False):
        if bias:
            vector[0] = 1

        scores = np.zeros(len(self.classLabels))

        for label in range(len(self.classLabels)):
            if numWeight is None:
                #logging.info('Using only label')
                w = self.weights[label]
            else:
                #logging.info('Using label and numWeight')
                w = self.weights[label][numWeight]

            score = np.dot(w,vector)
            scores[label] = score
        logging.debug('scores[%d,%d,%d] argmax=%d so nlabel is %s', scores[0], scores[1], scores[2], np.argmax(scores), self.classLabels[np.argmax(scores)])
        return self.classLabels[np.argmax(scores)]

    def updateWeights(self, numWeight, vector, correctLabel, wrongLabel=None):
        #logging.debug('%d elements to change',np.count_nonzero(vector))
        nonzero = np.nonzero(vector)
        if numWeight+1 == np.shape(self.weights)[self.weightsIdx]:
            nextWeight = 0
        else:
            nextWeight = numWeight + 1

		# each label
        for label in range(len(self.classLabels)):
            logging.debug('processing label %s',self.classLabels[label])
            # increase the right one
            if correctLabel == self.classLabels[label]:
                logging.debug('increasing %s',self.classLabels[label])
				# update weights
                self.weights[label][nextWeight] = self.weights[label][numWeight] + vector
            # decrease the wrong one
            if wrongLabel == self.classLabels[label]:
                logging.debug('decreasing %s',self.classLabels[label])
                # update weights
                self.weights[label][nextWeight] = self.weights[label][numWeight] - vector

    def copyWeights(self, numWeight):
        if numWeight+1 == np.shape(self.weights)[self.weightsIdx]:
            nextWeight = 0
        else:
            nextWeight = numWeight + 1

        for label in range(len(self.classLabels)):
            self.weights[label][nextWeight] = self.weights[label][numWeight]

    def train(self, samples, labels, repetitions = 1, randomOrder = False, 
				average = False):
        logging.info('Training')
        numSamples = np.shape(samples)[0]
        numFeatures = np.shape(samples)[1]

        self.weights = np.zeros((len(self.classLabels), numSamples, numFeatures))
        logging.info('will repeat %d',repetitions)
        c = 1
        upd = 0
        samplesIndices = range(numSamples)
		# seeding RandomState with zero so randoms are generated in the same way on each test
        np.random.seed(0)
		
        for rep in range(repetitions):
            if randomOrder:
                np.random.shuffle(samplesIndices)
            logging.info('=' * 50)
            logging.info('will traverse %d indices',len(samplesIndices))
            #logging.info(samplesIndices)
            numWeight = 0
            for numSample in samplesIndices:
                nLabel = self.predictLabel(samples[numSample], numWeight)
                logging.debug('smp %d label %s',numSample, labels[numSample])
                if nLabel != labels[numSample]:
                    if numWeight < numSamples:
                        self.updateWeights(numWeight, samples[numSample], labels[numSample], nLabel)
                        upd += 1
                    else:
                        logging.info('Omited last+1 update')
                else:
                    if numWeight < numSamples:
                        self.copyWeights(numWeight)
                    else:
                        logging.info('Omited last+1 copy')
                numWeight += 1
            c += 1
        logging.info('total updates %d',upd)
        logging.info('=' * 50)

        finalWeights = []
        for label in range(len(self.classLabels)):
            if average:
                # avg all weights for each class
                finalWeights.append(np.mean(self.weights[label],0))
            else:
                # last weight vectors
                finalWeights.append(self.weights[label][numSamples-1])
        return finalWeights

    


############################## EARTH MOVERS DISTANCE
## extras for word movers sample
#import os
#import numpy as np
#from sklearn.datasets import fetch_20newsgroups
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.cross_validation import train_test_split

# load word2vec trained model
#w2vModel = Word2Vec.load_word2vec_format("./GoogleNews-vectors-negative300.bin", binary=True)
# fp?
#fp = np.memmap("./data_embed.dat", dtype=np.double, mode='w+', shape=w2vModel.syn0norm.shape)
#fp[:] = w2vModel.syn0norm[:]
#with open("./data_embed.vocab", "w") as f:
#	for _, w in sorted((voc.index, word) for word, voc in wv.vocab.items()):
#		f.write(w)
#	del fp, wv 
# ???? why

#W = np.memmap("./data_embed.dat", dtype=np.double, mode="r", shape=(3000000, 300))
#with open("./data_embed.vocab") as f:
#    vocab_list = map(str.strip, f.readlines())

#vocab_dict = {w: k for k, w in enumerate(vocab_list)}
############################## EARTH MOVERS DISTANCE

# word movers distance
#def wmd(s1, s2):
#	d = 0;
#	vect = CountVectorizer(stop_words="english").fit([s1, s2])
#	#print("Features:",  ", ".join(vect.get_feature_names()))
#
#	from scipy.spatial.distance import cosine
#	v_1, v_2 = vect.transform([d1, d2])
#	v_1 = v_1.toarray().ravel()
#	v_2 = v_2.toarray().ravel()
#	print(v_1, v_2)
#	print("cosine(doc_1, doc_2) = {:.2f}".format(cosine(v_1, v_2)))
#
#	from sklearn.metrics import euclidean_distances
#	W_ = W[[vocab_dict[w] for w in vect.get_feature_names()]]
#	D_ = euclidean_distances(W_)
#	#print("d(addresses, speaks) = {:.2f}".format(D_[0, 7]))
#	#print("d(addresses, chicago) = {:.2f}".format(D_[0, 1]))
#
#	from pyemd import emd
#	# pyemd needs double precision input
#	v_1 = v_1.astype(np.double)
#	v_2 = v_2.astype(np.double)
#	v_1 /= v_1.sum()
#	v_2 /= v_2.sum()
#	D_ = D_.astype(np.double)
#	D_ /= D_.max()  # just for comparison purposes
#	print("d(doc_1, doc_2) = {:.2f}".format(emd(v_1, v_2, D_)))
#
#	return d
