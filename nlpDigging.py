# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 10:27:36 2016

@author: camg
"""
import logging
import numpy as np
import nltk
from nltk.corpus import stopwords
import spacy
import string
from collections import Counter
import SheffNLP
from SheffUtils import PlainTextReader,Document

def loadSamples(dirsLabels, textReader, docStore, nlpEngine, limit):
    """
    Return an index of samples-tags and a DocumentStore with all samples loaded
    """
    samplesIdx = {}
    # load samples, create a vector space representation of each
    for directory,label in dirsLabels.iteritems():
        logging.info('%s : %d', directory,label)
        for content in textReader.readTexts(directory,limit):
            doc = Document(content[:][:])
            doc.setTokens(nlpEngine.tokenise(doc.text))
            docId = docStore.addDocument(doc)
            samplesIdx[docId] = label
    return samplesIdx,docStore

def loadTests(dirsLabels, textReader, docStore, nlpEngine, limit, offset):
    """
    Return an index of samples-tags and a DocumentStore with all samples loaded
    """
    samplesIdx = {}
    tests = {}
    idx = 0
    # load samples, create a vector space representation of each
    for directory,label in dirsLabels.iteritems():
        logging.info('%s : %d', directory,label)
        for content in textReader.readTexts(directory,limit, offset):
            doc = Document(content[:][:])
            #doc.setTokens(nlpEngine.tokenise(doc.text))
            tokens = nlpEngine.tokenise(doc.text)
            # docId = docStore.addDocument(doc)
            samplesIdx[idx] = label
            tests[idx] = docStore.vectorize(tokens)
            idx += 1
    return samplesIdx,tests

def displayInfo(w):
    logging.info(np.shape(w))
    for n in range(np.shape(w)[0]):
        logging.info('w[%d] max %d min %d sum %d',n,np.argmax(w[n]),np.argmin(w[n]),np.sum(w[n]))
        logging.debug(w[n])

def main():
    logging.info('Started')

    labels = [-1,1]
    directories = {'./neg':labels[0],'./pos':labels[1]}

    logging.info('Loading NLP engine')
    nlpEngine  = SheffNLP.NlpEngine()
    docStore   = SheffNLP.DocumentStore()
    textReader = PlainTextReader()
    #perceptron1 = SheffNLP.Perceptron()
    perceptron2 = SheffNLP.Perceptron(binary=False,classLabels=labels)
    perceptrons = []
    perceptrons.append(perceptron2)
    #perceptrons.append(perceptron1)

    logging.info('Loading Samples')
    samplesLabels, docStore = loadSamples(directories, textReader, docStore, nlpEngine, 800)

    # docs should be represented in a dense matrix
    logging.info('Generating dense matrix')
    samples = docStore.generateMatrix()

    logging.info('total docs: %d total words: %d',docStore.totalDocuments(),docStore.totalWords())
    logging.info('matrix dimmensions: %d x %d',np.shape(samples)[0],np.shape(samples)[1])

    docStore.freeTraingingData()

    for perceptron in perceptrons:
        #logging.info('Strategy 1 - 1 pass, non-random')
        #w1 = perceptron.train(samples, samplesLabels)
        #logging.info('Strategy 2 - multiple passes, non-random')
        #w2 = perceptron.train(samples, samplesLabels, 10)
        #logging.info('Strategy 3 - 1 pass, random ordering')
        #w3 = perceptron.train(samples, samplesLabels, randomOrder=True)
        #logging.info('Strategy 4 - average weights')
        #w4 = perceptron.train(samples, samplesLabels, average = True)
        logging.info('Strategy 5 - multiple passes, random ordering')
        w5 = perceptron.train(samples, samplesLabels, 15, True)
        logging.info('Done training')

        #displayInfo(w1)
        #displayInfo(w2)
        #displayInfo(w3)
        #displayInfo(w4)
        #displayInfo(w5)

        logging.info('Loading test samples')
        testsLabels, tests = loadTests(directories, textReader, docStore, nlpEngine, 200, 800)

        #c,w = perceptron.test(w1, testsLabels, tests)
        #logging.info('Strategy 1 - correct: %d, wrong: %d, accuracy = %f',c,w,c/400.)
        #del(w1)
        #c,w = perceptron.test(w2, testsLabels, tests)
        #logging.info('Strategy 2 - correct: %d, wrong: %d, accuracy = %f',c,w,c/400.)
        #del(w2)
        #c,w = perceptron.test(w3, testsLabels, tests)
        #logging.info('Strategy 3 - correct: %d, wrong: %d, accuracy = %f',c,w,c/400.)
        #del(w3)
        #c,w = perceptron.test(w4, testsLabels, tests)
        #logging.info('Strategy 4 - correct: %d, wrong: %d, accuracy = %f',c,w,c/400.)
        #del(w4)
        c,w = perceptron.test(w5, testsLabels, tests)
        logging.info('Strategy 5 - correct: %d, wrong: %d, accuracy = %f',c,w,c/400.)
        logging.info(np.shape(w5))
        # top positively weighted features for each class
        n = 10
        if np.shape(w5)[0] > 1 :
            logging.info('MC shape(w5)=%dx%d',np.shape(w5)[0],np.shape(w5)[1])
            # multi-class perceptron
            top10neg = docStore.topNwords(w5[0],n,reverse = True)
            top10pos = docStore.topNwords(w5[1],n,reverse = True)
        else:
            logging.info('len(w5)=%d',len(w5))
            top10neg = docStore.topNwords(w5[0],n,reverse = False)
            top10pos = docStore.topNwords(w5[0],n,reverse = True)

        del(w5)

        logging.info('Top %d negative words:',n)
        for row in top10neg:
            logging.info('%s %d',row[0],row[1])
        logging.info('Top %d positive words:',n)
        for row in top10pos:
            logging.info('%s %d',row[0],row[1])

        logging.info('Finished')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-4s %(message)s', filename='lab2.log')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logging.getLogger('').addHandler(ch)

    main()
