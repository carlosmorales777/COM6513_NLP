import logging
import numpy as np
import nltk
from nltk.corpus import stopwords
import spacy
import string
from collections import Counter

STOPWORDS = set(nltk.corpus.stopwords.words('english') + ["n't","'s","&nbsp"])
SYMBOLS = " ".join(string.punctuation).join("\\").split(" ")

class NlpEngine:
    def __init__(self):
        self.nlp = spacy.en.English()

    def tokenise(self, text):
        ntext = self.nlp(text)

        tokens = []
        for toke in ntext:
            tokens.append(toke.lemma_.lower().strip() if toke.lemma_ != "-PRON-" else toke.lower_)

        # stop words
        tokens = [token for token in tokens if token not in STOPWORDS]

        # stop symbols
        tokens = [token for token in tokens if token not in SYMBOLS]

        while "\\n" in tokens:
            tokens.remove('\\n')

        return tokens

class DocumentStore:
    def __init__(self):
        self.words = {}
        self.wordId = 0
        self.docId = 0
        self.documents = {}

    def addDocument(self, document):
        doc = Counter()
        for word in document.getTokens():
            if word not in self.words:
                self.words[word] = self.wordId
                self.wordId += 1
            doc[self.words[word]] += 1
        # store document
        self.documents[self.docId] = doc
        self.docId += 1
        return self.docId - 1

    def totalWords(self):
        return self.wordId

    def printWords(self):
        print self.words

    def totalDocuments(self):
        return self.docId

    def generateMatrix(self):
        """
        each row represents a document, each column map to one word from our master word index,
        each document word count is stored in its corresponding cell accordingly
        """
        m = np.zeros((self.docId, self.wordId))
        for docId, doc in self.documents.iteritems():
            for wordId,v in doc.iteritems():
                m[docId,wordId] = v
        return m

    def vectorize(self, tokens):
        vector = np.zeros(self.wordId)
        for word in tokens:
            if word in self.words:
                vector[self.words[word]] += 1
        return vector

    def topNwords(self, vector, n, reverse = False):
        logging.info(vector)
        wordsWeights = []
        for wrd,idx in self.words.iteritems():
            # store index, word, weight
            wordsWeights.append([wrd,vector[idx]])
        # sort
        topWs = sorted(wordsWeights,key=lambda ww:ww[1], reverse=reverse)[0:n]
        # find top 10
        return topWs

    def freeTraingingData(self):
        del(self.documents)
        self.documents = {}

class Perceptron:
    def __init__(self, binary=True, classLabels=None):
        self.binary = binary
        if binary:
            self.weightsIdx = 0
        else:
            self.weightsIdx = 1
            self.classLabels = classLabels

    def predictLabel(self, vector, numWeight = None, bias = False):
        if bias:
            vector[0] = 1

        if self.binary:
            if numWeight is not None:
                w = self.weights[numWeight]
            else:
                w = self.weights
            sign = np.sign(np.dot(w,vector))
            return sign
        else:
            scores = np.zeros(len(self.classLabels))

            for label in range(len(self.classLabels)):
                if numWeight is not None:
                    w = self.weights[label][numWeight]
                else:
                    w = self.weights[label]

                score = np.dot(w,vector)
                scores[label] = score
                logging.debug('scores[%d,%d] argmax=%d so nlabel is %d',scores[0],scores[1],np.argmax(scores),self.classLabels[np.argmax(scores)])
            return self.classLabels[np.argmax(scores)]

    def updateWeights(self,numWeight,vector,correctLabel,wrongLabel=None):
        logging.debug('%d elements to change',np.count_nonzero(vector))
        nonzero = np.nonzero(vector)
        if numWeight+1 == np.shape(self.weights)[self.weightsIdx]:
            nextWeight = 0
        else:
            nextWeight = numWeight + 1
        if self.binary:
            self.weights[nextWeight] = self.weights[numWeight] + correctLabel * vector
            logging.debug('new values')
            logging.debug(self.weights[nextWeight][nonzero])
        else:
            for label in range(len(self.classLabels)):
                # increase the right one
                if correctLabel == self.classLabels[label]:
                    logging.debug('correctLabel %d',correctLabel)
                    self.weights[label][nextWeight] = self.weights[label][numWeight] + vector
                    logging.debug('new values')
                    logging.debug('substracting vector from new values (must be zero) = %d',np.sum(self.weights[label][nextWeight][nonzero] - vector[nonzero]))
                # decrease the wrong one
                if wrongLabel == self.classLabels[label]:
                    logging.debug('wrongLabel %d',wrongLabel)
                    self.weights[label][nextWeight] = self.weights[label][numWeight] - vector
                    logging.debug('adding vector to new values (must be zero) = %d',np.sum(self.weights[label][nextWeight][nonzero] + vector[nonzero]))

    def copyWeights(self, numWeight):
        if numWeight+1 == np.shape(self.weights)[self.weightsIdx]:
            nextWeight = 0
        else:
            nextWeight = numWeight + 1

        if self.binary:
            self.weights[nextWeight] = self.weights[numWeight]
        else:
            for label in range(len(self.classLabels)):
                self.weights[label][nextWeight] = self.weights[label][numWeight]

    def train(self, samples, labels, repetitions = 1, randomOrder = False, average = False):
        logging.info('Training')
        numSamples = np.shape(samples)[0]
        numWords = np.shape(samples)[1]

        if self.binary:
            self.weights = np.zeros((numSamples, numWords))
        else:
            self.weights = np.zeros((len(self.classLabels), numSamples, numWords))
        logging.info('will repeat %d',repetitions)
        c = 1
        upd = 0
        samplesIndices = range(numSamples)
        for rep in range(repetitions):
            if randomOrder:
                np.random.shuffle(samplesIndices)
            logging.info('=' * 50)
            logging.info('will traverse %d indices',len(samplesIndices))
            numWeight = 0
            for numSample in samplesIndices:
                nLabel = self.predictLabel(samples[numSample], numWeight)
                logging.debug('Sample %4d label is %d',numSample,labels[numSample])
                if nLabel != labels[numSample]:
                    if numWeight < numSamples:
                        if self.binary:
                            self.updateWeights(numWeight,samples[numSample],labels[numSample])
                        else:
                            self.updateWeights(numWeight,samples[numSample],labels[numSample],nLabel)
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
        if self.binary:
            if average:
                # avg all weights for each class
                finalWeights.append(np.mean(self.weights,0))
            else:
                # last weight vectors
                finalWeights.append(self.weights[numSamples-1])
        else:
            for label in range(len(self.classLabels)):
                if average:
                    # avg all weights for each class
                    finalWeights.append(np.mean(self.weights[label],0))
                else:
                    # last weight vectors
                    finalWeights.append(self.weights[label][numSamples-1])
        return finalWeights

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
