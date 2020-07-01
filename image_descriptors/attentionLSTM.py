import numpy as np
import sys, torch

sys.path.append("../")

from models.encoders import EncoderAttention
from models.decoders.Visual_attention.Attention_decoder import AttentionDecoder


def attention_lstm(vocab_size):

    encoder = EncoderAttention()
    decoder = AttentionDecoder(
        512,
        512,
        512,
        vocab_size,
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    return encoder, decoder
