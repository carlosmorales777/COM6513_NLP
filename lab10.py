from SheffUtils import JsonReader
from nltk.translate import bleu_score
from spacy.en import English

reader = JsonReader('./snli_1.0/snli_1.0_dev.jsonl')
nlp = English()

noLabel = 0.
correctLabel = 0.
wrongLabel = 0.

for snt in reader:
    if snt['gold_label'] != '-':
        s1 = nlp(snt['sentence1'])
        s2 = nlp(snt['sentence2'])

        a = bleu_score.sentence_bleu(s1[:].text, s2[:].text)
        b = s1.similarity(s2)
        c = float(min(len(s1),len(s2)))/float(max(len(s1),len(s2)))
        avgScore = (a+b+c)/3.
        clss = 'contradiction' if avgScore < 0.3 else 'entailment' if avgScore > 0.6 else 'neutral'

        if clss == snt['gold_label']:
            correctLabel += 1.
        else:
            wrongLabel += 1.
    else:
        noLabel += 1.
total = float(correctLabel + wrongLabel + noLabel)
print('correctLabel=%d wrongLabel=%d noLabel=%d total=%d, accy=%f'%(correctLabel, wrongLabel, noLabel,total,correctLabel/total))
