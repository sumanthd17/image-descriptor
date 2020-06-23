import numpy as np
import sys

sys.path.append("../")

from utils.dataloader import dataloader
from utils.transforms import transform_train, transform_val

from models.encoders import VisualBackbone
from models.decoders import TextualHeadLSTM

from .train import train


def trainLSTM(args):
    data_loader = dataloader(
        transform=transform_train,
        mode="train",
        batch_size=args.batch_size,
        vocab_threshold=args.vocab_threshold,
        from_vocab_file=args.from_vocab_file,
        vocab_file="./vocab.pkl",
        data_path=args.data_dir,
        image_data_unavailable=args.data_unavailable,
    )

    vocab_size = len(data_loader.dataset.vocab)
    print("vocabulary size: {}".format(vocab_size))

    encoder = VisualBackbone(300)
    decoder = TextualHeadLSTM(300, 512, vocab_size, num_layers=1)

    train(encoder, decoder, data_loader, vocab_size, args)
