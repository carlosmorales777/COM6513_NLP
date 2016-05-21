from os import listdir
from os.path import isfile, join
import json

class PlainTextReader:
    def __init__(self):
        pass

    def readContent(self, filePath):
        with open(filePath, 'r') as inputfile:
            return inputfile.readlines()

    def readTexts(self, directory, limit = None, start = 0):
        files = [f for f in listdir(directory) if isfile(join(directory,f)) and f.endswith('.txt')]
        files.sort()

        contents = []
        for fil in files[start:start+limit]:
            contents.append([self.readContent(join(directory,fil))])
        return contents

class Document:
    def __init__(self, sentence1, sentence2, gold_label = ''):
        # spacy Doc
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.gold_label = unicode(gold_label)

    def setTokens(self, tokens):
        # simple list of tokens
        self.tokens = tokens

    def getTokens(self):
        return self.tokens

    def setTokens2(self, tokens):
        # simple list of tokens
        self.tokens2 = tokens

    def getTokens2(self):
        return self.tokens2 

class JsonReader:
    """
    Iterator class to load large JSON files line-by-line into memory
    Input:
        jsonFile txt file to read
        lastLine line number to start reading from
    Output:
        json object
    """
    def __init__(self, jsonFile, lastLine = -1):
        self.jsonFile = jsonFile
        self.lastLine = lastLine
        self.ln = 0
        self.jfile = open(self.jsonFile, 'r')

    def __iter__(self):
        return self

    def next(self):
        for line in self.jfile:
            if self.ln > self.lastLine:
                jdata = json.loads(line)
                return jdata
            self.ln += 1
        else:
            self.jfile.close()
            raise StopIteration()

