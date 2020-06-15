import os
import pickle
from pycocotools.coco import COCO
import nltk
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from collections import Counter

nltk.download("punkt")


class Vocabulary:
    def __init__(self, threshold, vocab_file, from_vocab_file, annotation_file):
        self.start_word = "<start>"
        self.end_word = "<end>"
        self.unk_word = "<unk>"
        self.threshold = threshold
        self.vocab_file = vocab_file
        self.from_vocab_file = from_vocab_file
        self.annotation_file = annotation_file

        self.get_vocab()

    def get_vocab(self):
        if os.path.exists(self.vocab_file) and self.from_vocab_file:
            with open(self.vocab_file, "rb") as f:
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
        else:
            self.build_vocab()
            with open(self.vocab_file, "wb") as f:
                pickle.dump(self, f)

    def build_vocab(self):
        self.init_vocab()
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_captions()

    def init_vocab(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_captions(self):
        coco = COCO(self.annotation_file)
        ids = coco.anns.keys()

        counter = Counter()
        for i, id in tqdm(enumerate(ids)):
            caption = str(coco.anns[id]["caption"])
            tokens = word_tokenize(caption.lower())

            counter.update(tokens)

        words = [key for key, val in counter.items() if val > self.threshold]

        for word in words:
            self.add_word(word)

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx[self.unk_word]
        else:
            return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
