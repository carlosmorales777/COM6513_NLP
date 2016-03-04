from os import listdir
from os.path import isfile, join

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
    def __init__(self, text):
        self.text = unicode(text)

    def setTokens(self, tokens):
        self.tokens = tokens

    def getTokens(self):
        return self.tokens

