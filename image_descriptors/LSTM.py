import numpy as np
import sys

sys.path.append("../")

from models.encoders import ResNet50
from models.decoders.LSTM import TextualHeadLSTM


def LSTM(vocab_size):

    encoder = ResNet50(300)
    decoder = TextualHeadLSTM(300, 512, vocab_size, num_layers=1)

    return encoder, decoder
