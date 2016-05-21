import logging
from spacy.en import English
import SheffUtils # import JsonReader
import SheffNLP # import NlpEngine, W2V, PerceptronEntailment

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-4s %(message)s', filename='finalEntailment.log')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logging.getLogger('').addHandler(ch)

# now we load the spacy models
nlp = English()

reader = SheffUtils.JsonReader('./snli_1.0/snli_1.0_dev.jsonl')
nlpEng = SheffNLP.NlpEngine()
docStore   = SheffNLP.DocumentStore(tokenWeights = False, extraFeatures = True,
    EXTRA_WEIGHTS_LABELS = [
        #'bleuScore',
        'similarityScore',
        #'wordMoversDistance',
        #'crossUnigramsRatio'
        ])

perceptron = SheffNLP.PerceptronEntailment(['neutral','contradiction','entailment'])

counter = 0
# load the training files
samplesLabels= {}
for snt in reader:
    if '-' != snt['gold_label']:
        s1 = nlp(snt['sentence1'].lower()) # gensim Docs
        s2 = nlp(snt['sentence2'].lower())
        tk1 = nlpEng.tokenise(snt['sentence1'].lower()) # basic token list, not gensim
        tk2 = nlpEng.tokenise(snt['sentence2'].lower())
        doc = SheffUtils.Document(s1, s2, snt['gold_label']) # gensim & string
        doc.setTokens(tk1)
        doc.setTokens2(tk2)
        docId = docStore.addDocument(doc)
        samplesLabels[docId] = snt['gold_label']

samples = docStore.generateMatrix()
docStore.releaseTraingingData()

weights = perceptron.train(samples, samplesLabels, 10, True)

logging.info('Done training weigths, generated: %d weights'%(len(weights)))

# load and test
reader = SheffUtils.JsonReader('./snli_1.0/snli_1.0_test.jsonl')

testsLabels= {}
tests = {}
idx = 0
for snt in reader:
    if '-' != snt['gold_label']:
        s1 = nlp(snt['sentence1']) # gensim Docs
        s2 = nlp(snt['sentence2'])
        tk1 = nlpEng.tokenise(snt['sentence1']) # basic token list, not gensim
        tk2 = nlpEng.tokenise(snt['sentence2'])
        tests[idx] = docStore.vectorize(s1, s2)    
        testsLabels[idx] = snt['gold_label']
        idx += 1

c, w = perceptron.test(weights, testsLabels, tests)

logging.info('Correct: %d, wrong: %d, accuracy = %f',c,w, float(c) / float(c + w))



