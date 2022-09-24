import os
from io import open
import torch
import json


class Corpus(object):
    def __init__(self, path):
        self.dictionaryen = self.readjason(os.path.join(path, 'en.BPE.txt.json'))
        self.dictionaryha = self.readjason(os.path.join(path, 'ha.BPE.txt.json'))
        self.english = self.TtoIen(os.path.join(path, 'en.BPE.txt'))
        self.hausa = self.TtoIha(os.path.join(path, 'ha.BPE.txt'))

    def readjason(self, path):
        with open(path, encoding='utf-8') as f:
            vocab = json.load(f)
        return vocab

    def TtoIen(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # convert word to indices
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<EOS>']
                ids = []
                for word in words:
                    if word not in self.dictionaryen:
                        ids.append(2)
                    else:
                        ids.append(self.dictionaryen[word])

                idss.append(torch.tensor(ids).type(torch.int64))


        return idss


    def TtoIha(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # convert word to indices
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<EOS>']
                ids = []
                for word in words:
                    if word not in self.dictionaryha:
                        ids.append(2)
                    else:
                        ids.append(self.dictionaryha[word])

                idss.append(torch.tensor(ids).type(torch.int64))


        return idss





class Corpus_val(object):
    def __init__(self, path):
        self.dictionaryen_val = self.readjason(os.path.join(path, 'en_val.BPE.txt.json'))
        self.dictionaryha_val = self.readjason(os.path.join(path, 'ha_val.BPE.txt.json'))
        self.english_val = self.TtoIen(os.path.join(path, 'en_val.BPE.txt'))
        self.hausa_val = self.TtoIha(os.path.join(path, 'ha_val.BPE.txt'))

    def readjason(self, path):
        with open(path, encoding='utf-8') as f:
            vocab = json.load(f)
        return vocab

    def TtoIen(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # convert word to indices
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<EOS>']
                ids = []
                for word in words:
                    if word not in self.dictionaryen_val:
                        ids.append(2)
                    else:
                        ids.append(self.dictionaryen_val[word])

                idss.append(torch.tensor(ids).type(torch.int64))


        return idss


    def TtoIha(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # convert word to indices
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<EOS>']
                ids = []
                for word in words:
                    if word not in self.dictionaryha_val:
                        ids.append(2)
                    else:
                        ids.append(self.dictionaryha_val[word])

                idss.append(torch.tensor(ids).type(torch.int64))


        return idss