import sys
from bert_tagger.utils.alphabet import Alphabet
from bert_tagger.utils.functions import *


class Data:
    def __init__(self):
        self.label_alphabet = Alphabet("label", unkflag=False, padflag=False)
        self.tagscheme = "BIO"
        self.train_ids = []
        self.dev_ids = []
        self.test_ids = []
        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []

    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("     Label        Scheme: %s" % (self.tagscheme))
        print("     Label Alphabet Size: %s" % self.label_alphabet.size())
        print("     Train instance number: %s" % (len(self.train_ids)))
        print("     Dev   instance number: %s" % (len(self.dev_ids)))
        print("     Test  instance number: %s" % (len(self.test_ids)))
        print("DATA SUMMARY END.")
        sys.stdout.flush()

    def generate_instance(self, input_doc, name, bert_file):
        ids, texts = read_instance(input_doc, self.label_alphabet, bert_file)
        if name == "train":
            self.train_ids = ids
            self.train_texts = texts
        elif name == "dev":
            self.dev_ids = ids
            self.dev_texts = texts
        elif name == "test":
            self.test_ids = ids
            self.test_texts = texts
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s" % name)

    def get_tag_scheme(self):
        startS = False
        startB = False
        for label, _ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagscheme = "BMES"
            else:
                self.tagscheme = "BIO"