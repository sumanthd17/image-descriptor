import argparse
import os
import sys

from utils.dataloader import dataloader
from utils.transforms import transform_train, transform_val
from utils.argparsers import str2bool

from image_descriptors.LSTM import LSTM
from training import train


parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--data_dir", dest="data_dir", default="./data", help="path of the dataset",
)
parser.add_argument(
    "--model_dir",
    dest="model_dir",
    default="./save_model",
    help="path of the model checkpoints",
)
parser.add_argument(
    "--data_unavailable",
    dest="data_unavailable",
    default=True,
    type=str2bool,
    help="set to True if Image data not available locally. \
        If set to True then internet connection is a much for \
        getting image data from 'http://images.cocodataset.org/'",
)
parser.add_argument(
    "--mode", dest="mode", default="train", help="set mode to either train or val"
)
parser.add_argument("--epochs", dest="epochs", type=int, default=1, help="# epoch")
parser.add_argument(
    "--batch_size",
    dest="batch_size",
    type=int,
    default=64,
    help="# examples per batch",
)
parser.add_argument(
    "--save_every",
    dest="save_every",
    type=int,
    default=1,
    help="# epochs after which the model has to be saved",
)
parser.add_argument(
    "--print_every",
    dest="print_every",
    type=int,
    default=100,
    help="# of steps after which the model performance can be printed",
)
parser.add_argument(
    "--from_vocab_file",
    dest="from_vocab_file",
    default=True,
    type=str2bool,
    help="set to False if vocabulary has to created i.e., vocab.pkl isn't available",
)
parser.add_argument(
    "--vocab_file",
    dest="vocab_file",
    default="./vocab.pkl",
    help="path to vocab file. Will be considered only if from_vocab_file is True",
)
parser.add_argument(
    "--vocab_threshold", dest="vocab_threshold", default=5, type=int, help="",
)
parser.add_argument(
    "--model", dest="model", default="lstm", help="decoder model to be used"
)
parser.add_argument(
    "--cont_train",
    dest="cont_train",
    type=str2bool,
    default=False,
    help="choose where to train from starting or continue from latest checkpoint available.",
)

args = parser.parse_args()

if not os.path.exists(os.path.join(args.model_dir, args.model)):
    os.makedirs(os.path.join(args.model_dir, args.model))

transformer = transform_train if (args.mode == "train") else transform_val

data_loader = dataloader(
    transform=transformer,
    mode=args.mode,
    batch_size=args.batch_size,
    vocab_threshold=args.vocab_threshold,
    from_vocab_file=args.from_vocab_file,
    vocab_file=args.vocab_file,
    data_path=args.data_dir,
    image_data_unavailable=args.data_unavailable,
)

vocab_size = len(data_loader.dataset.vocab)
print("vocabulary size: {}".format(vocab_size))

if args.model == "lstm":
    encoder, decoder = LSTM(vocab_size)

if args.mode == "train":
    train(encoder, decoder, data_loader, vocab_size, args)
