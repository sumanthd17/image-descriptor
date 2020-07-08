import math
import time
import sys
import os

import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np

from train import train
from validate import validate

from utils.transforms import transform_train, transform_val
from utils.dataloader import dataloader

from models.encoders import ResNet50
from models.decoders import TextualHead


def early_stopping(val_bleus, patience=3):
    # The number of epochs should be at least patience before checking
    # for convergence
    if patience > len(val_bleus):
        return False
    latest_bleus = val_bleus[-patience:]
    # If all the latest Bleu scores are the same, return True
    if len(set(latest_bleus)) == 1:
        return True
    max_bleu = max(val_bleus)
    if max_bleu in latest_bleus:
        # If one of recent Bleu scores improves, not yet converged
        if max_bleu not in val_bleus[: len(val_bleus) - patience]:
            return False
        else:
            return True
    # If none of recent Bleu scores is greater than max_bleu, it has converged
    return True


train_losses = []
val_losses = []
val_bleus = []
best_val_bleu = float("-INF")

BATCH_SIZE = 32

train_loader = dataloader(
    mode="val",
    transform=transform_val,
    batch_size=BATCH_SIZE,
    vocab_threshold=5,
    vocab_file="./pickle/vocab.pkl",
    from_vocab_file=True,
    img_dir_path="./data/val2014",
    captions_path="./data/annotations/captions_val2014.json",
)

val_loader = dataloader(
    mode="val",
    transform=transform_val,
    batch_size=BATCH_SIZE,
    vocab_threshold=5,
    vocab_file="./pickle/vocab.pkl",
    from_vocab_file=True,
    img_dir_path="./data/val2014",
    captions_path="./data/annotations/captions_val2014.json",
)

total_train_step = math.ceil(
    len(train_loader.dataset.caption_lengths) / train_loader.batch_size
)

total_val_step = math.ceil(
    len(val_loader.dataset.caption_lengths) / val_loader.batch_size
)

NUM_EPOCHS = 10
VOCAB_SIZE = len(train_loader.dataset.vocab)

visual = ResNet50(1024)
textual = TextualHead(VOCAB_SIZE, 1024, 1, 16, 4096, 0.1, 30, 0)

criterion = nn.CrossEntropyLoss()

params = list(visual.parameters()) + list(textual.parameters())
optimizer = torch.optim.Adam(params=params, lr=0.01)

if torch.cuda.is_available():
    visual.cuda()
    textual.cuda()
    criterion.cuda()

start_time = time.time()
for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = train(
        train_loader,
        visual,
        textual,
        criterion,
        optimizer,
        VOCAB_SIZE,
        epoch,
        total_train_step,
        BATCH_SIZE,
    )
    train_losses.append(train_loss)

    val_loss, val_bleu = validate(
        val_loader,
        visual,
        textual,
        criterion,
        train_loader.dataset.vocab,
        epoch,
        total_val_step,
        BATCH_SIZE,
    )
    val_losses.append(val_loss)
    val_bleus.append(val_bleu)

    if val_bleu > best_val_bleu:
        print(
            "Validation Bleu-4 improved from {:0.4f} to {:0.4f}, saving model to best-model.pkl".format(
                best_val_bleu, val_bleu
            )
        )
        best_val_bleu = val_bleu
        filename = os.path.join("./weights", "best-model.pkl")
        torch.save(
            {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_bleu": val_bleu,
                "val_bleus": val_bleus,
                "epoch": epoch,
            },
            filename,
        )
    else:
        print(
            "Validation Bleu-4 did not improve, saving model to model-{}.pkl".format(
                epoch
            )
        )

    filename = os.path.join("./weights", "model-{}.pkl".format(epoch))
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_bleu": val_bleu,
            "val_bleus": val_bleus,
            "epoch": epoch,
        },
        filename,
    )

    print("Epoch [%d/%d] took %ds" % (epoch, NUM_EPOCHS, time.time() - start_time))
    if epoch > 5:
        if early_stopping(val_bleus, 3):
            break

    start_time = time.time()
