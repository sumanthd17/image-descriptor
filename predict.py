import argparse
import os, sys
import pickle

import PIL
import torch
from torch.autograd import Variable

from image_descriptors.LSTM import LSTM
from utils.clean_sentence import clean_sentence
from utils.load_checkpoint import load_checkpoint
from utils.transforms import transform_val


parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--vocab_file",
    dest="vocab_file",
    default="./vocab.pkl",
    help="path to vocab file, for loading vocabulary",
)
parser.add_argument(
    "--model_dir",
    dest="model_dir",
    default="./save_model",
    help="path of the model checkpoints",
)
parser.add_argument(
    "--model", dest="model", default="lstm", help="decoder model to be used"
)
parser.add_argument("--img_path", dest="img_path", help="Path to input image")
args = parser.parse_args()

if args.img_path == None:
    print("Image path is required...")
    sys.exit()

if not os.path.exists(args.vocab_file):
    print("Vocab file not available...")
    sys.exit()

with open(args.vocab_file, "rb") as f:
    vocab = pickle.load(f)

device = torch.device("cpu")

if args.model == "lstm":
    encoder, decoder = LSTM(len(vocab))

encoder, decoder, _, _ = load_checkpoint(encoder, decoder, None, device, args, False)
encoder.eval()
decoder.eval()

img = PIL.Image.open(args.img_path).convert("RGB")
img = transform_val(img).float()
img = img.unsqueeze_(0)
img = Variable(img)

visual_features = encoder(img)
output_sentence = decoder.sample(visual_features)

print(clean_sentence(output_sentence, vocab))
