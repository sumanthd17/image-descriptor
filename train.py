import math
import time
import sys
import os

import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np

from utils.transforms import transform_train, transform_val
from utils.dataloader import dataloader

from models.encoders import ResNet50
from models.decoders import TextualHead

PRINT_EVERY = 100


def train(
    train_loader,
    encoder,
    decoder,
    criterion,
    optimizer,
    vocab_size,
    epoch,
    total_step,
    start_step=1,
    start_loss=0.0,
):
    encoder.train()
    decoder.train()

    total_loss = start_loss

    start_train_time = time.time()

    for i_step in range(start_step, total_step + 1):
        indices = train_loader.dataset.get_indices()
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        train_loader.batch_sampler.sampler = new_sampler

        for batch in train_loader:
            images, captions = batch[0], batch[1]
            break

        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        features = encoder(images)
        outputs = decoder(features, captions, captions.shape[1] * torch.ones(32))

        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        stats = "Epoch %d, Train step [%d/%d], %ds, Loss: %.4f, Perplexity: %5.4f" % (
            epoch,
            i_step,
            total_step,
            time.time() - start_train_time,
            loss.item(),
            np.exp(loss.item()),
        )
        print("\r" + stats, end="")
        sys.stdout.flush()

        if i_step % PRINT_EVERY == 0:
            print("\r" + stats)
            filename = os.path.join(
                "./models", "train-model-{}{}.pkl".format(epoch, i_step)
            )
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "total_loss": total_loss,
                    "epoch": epoch,
                    "train_step": i_step,
                },
                filename,
            )
            start_train_time = time.time()

    return total_loss / total_step


dl = dataloader(
    mode="val",
    transform=transform_val,
    batch_size=32,
    vocab_threshold=5,
    vocab_file="./pickle/vocab.pkl",
    from_vocab_file=True,
    img_dir_path="./data/val2014",
    captions_path="./data/annotations/captions_val2014.json",
)
indices = dl.dataset.get_indices()
new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
dl.batch_sampler.sampler = new_sampler
total_val_step = math.ceil(len(dl.dataset.caption_lengths) / dl.batch_size)

visual = ResNet50(1024)
textual = TextualHead(len(dl.dataset.vocab), 1024, 1, 16, 4096, 0.1, 30, 0)

criterion = nn.CrossEntropyLoss()

params = list(visual.parameters()) + list(textual.parameters())
optimizer = torch.optim.Adam(params=params, lr=0.01)

print(len(dl.dataset.vocab))
train(
    dl, visual, textual, criterion, optimizer, len(dl.dataset.vocab), 1, total_val_step
)
