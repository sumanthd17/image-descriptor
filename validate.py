import math
import time
import sys
import os

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def word_list(word_idx_list, vocab):
    """Take a list of word ids and a vocabulary from a dataset as inputs
    and return the corresponding words as a list.
    """
    word_list = []
    for i in range(len(word_idx_list)):
        vocab_id = word_idx_list[i]
        word = vocab.idx2word[vocab_id]
        if word == vocab.end_word:
            break
        if word != vocab.start_word:
            word_list.append(word)
    return word_list


def validate(
    val_loader,
    encoder,
    decoder,
    criterion,
    vocab,
    epoch,
    total_step,
    batch_size,
    start_step=1,
    start_loss=0.0,
    start_bleu=0.0,
):
    PRINT_EVERY = 1000
    encoder.eval()
    decoder.eval()

    smoothing = SmoothingFunction()

    total_loss = start_loss
    total_bleu_4 = start_bleu

    start_val_time = time.time()

    with torch.no_grad():
        for i_step in range(start_step, total_step + 1):
            indices = val_loader.dataset.get_indices()
            new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
            val_loader.batch_sampler.sampler = new_sampler

            for batch in val_loader:
                images, captions = batch[0], batch[1]
                break

            caption_lengths = captions.shape[1] * torch.ones(batch_size)
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
                caption_lengths = caption_lengths.cuda()

            features = encoder(images)
            outputs = decoder(features, captions, caption_lengths)

            batch_bleu_4 = 0.0
            for i in range(len(outputs)):
                predicted_ids = []
                for scores in outputs[i]:
                    predicted_ids.append(scores.argmax().item())
                predicted_word_list = word_list(predicted_ids, vocab)
                caption_word_list = word_list(captions[i].cpu().numpy(), vocab)

                batch_bleu_4 += sentence_bleu(
                    [caption_word_list],
                    predicted_word_list,
                    smoothing_function=smoothing.method1,
                )
            total_bleu_4 += batch_bleu_4 / len(outputs)

            loss = criterion(outputs.view(-1, len(vocab)), captions.view(-1))
            total_loss += loss.item()

            stats = (
                "Epoch %d, Val step [%d/%d], %ds, Loss: %.4f, Perplexity: %5.4f, Bleu-4: %.4f"
                % (
                    epoch,
                    i_step,
                    total_step,
                    time.time() - start_val_time,
                    loss.item(),
                    np.exp(loss.item()),
                    batch_bleu_4 / len(outputs),
                )
            )

            print("\r" + stats, end="")
            sys.stdout.flush()

            if i_step % PRINT_EVERY == 0:
                print("\r" + stats)
                start_val_time = time.time()

        return total_loss / total_step, total_bleu_4 / total_step
