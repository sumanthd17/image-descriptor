import numpy as np
import sys

sys.path.append("../")

from models.encoders import VisualBackbone
from models.decoders import TextualHeadLSTM


def LSTM(vocab_size, args):

    encoder = VisualBackbone(300)
    decoder = TextualHeadLSTM(300, 512, vocab_size, num_layers=1)

    return encoder, decoder
