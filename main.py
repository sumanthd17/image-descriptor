import argparse
import os
import sys

from utils.dataloader import dataloader
from utils.transforms import transform_train, transform_val

from training.LSTM import trainLSTM

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
    help="set to False if vocabulary has to created i.e., vocab.pkl isn't available",
)
parser.add_argument(
    "--vocab_threshold", dest="vocab_threshold", default=5, type=int, help="",
)
parser.add_argument("--model", dest="model", help="decoder model to be used")

args = parser.parse_args()

if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)

transformer = transform_train if (args.mode == "train") else transform_val

data_loader = dataloader(
    transform=transformer,
    mode=args.mode,
    batch_size=args.batch_size,
    vocab_threshold=args.vocab_threshold,
    from_vocab_file=args.from_vocab_file,
    vocab_file="./vocab.pkl",
    data_path=args.data_dir,
    image_data_unavailable=args.data_unavailable,
)

if args.model == "lstm":
    if args.mode == "train":
        trainLSTM(data_loader, args)
