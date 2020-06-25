import numpy as np
import sys

from .training import train

sys.path.append("../")

from models.encoders import VisualBackbone
from models.decoders import TextualHeadLSTM


def trainLSTM(data_loader, args):

    vocab_size = len(data_loader.dataset.vocab)
    print("vocabulary size: {}".format(vocab_size))

    encoder = VisualBackbone(300)
    decoder = TextualHeadLSTM(300, 512, vocab_size, num_layers=1)

    train(encoder, decoder, data_loader, vocab_size, args)
