import numpy as np
import math, os, sys
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu

from utils.load_checkpoint import load_checkpoint


def validation(encoder, decoder, val_loader, vocab_size, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    decoder.to(device)

    encoder.eval()
    decoder.eval()

    criterion = (
        nn.CrossEntropyLoss().cuda()
        if torch.cuda.is_available()
        else nn.CrossEntropyLoss()
    )

    if args.mode == "val":
        encoder, decoder, _, _ = load_checkpoint(
            encoder, decoder, None, device, args, False
        )

    total_step = math.ceil(
        len(val_loader.dataset.caption_lengths) / val_loader.batch_sampler.batch_size
    )

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            imgs = encoder(imgs)

            if args.model == "lstm":
                scores = decoder(imgs, caps)
                loss = criterion(scores.view(-1, vocab_size), caps.view(-1))
            elif args.model == "attention":
                scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
                    imgs, caps, caplens
                )
                targets = caps_sorted[:, 1:]
                scores_copy = scores.clone()

                scores = pack_padded_sequence(
                    scores, decode_lengths, batch_first=True
                ).data
                targets = pack_padded_sequence(
                    targets, decode_lengths, batch_first=True
                ).data

                loss = criterion(scores, targets)
                loss += 1.0 * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

                scores = scores_copy

            stats = "Step [%d/%d], Loss: %.4f, Perplexity: %5.4f" % (
                i + 1,
                total_step,
                loss.item(),
                np.exp(loss.item()),
            )

            print("\r" + stats, end="")
            sys.stdout.flush()

            if (i + 1) % args.print_every == 0:
                print("\r" + stats)

            # References
            if args.model == "attention":
                allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(
                        lambda c: [
                            w
                            for w in c
                            if w
                            not in {
                                val_loader.dataset.vocab("<start>"),
                                val_loader.dataset.vocab("<pad>"),
                            }
                        ],
                        img_caps,
                    )
                )
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, _ in enumerate(preds):
                if args.model == "attention":
                    temp_preds.append(preds[j][: decode_lengths[j]])  # remove pads
                elif args.model == "lstm":
                    temp_preds.append(preds[j])
            hypotheses.extend(temp_preds)

            assert len(references) == len(hypotheses)

    bleu4 = corpus_bleu(references, hypotheses)
    print("\n\nBLEU-4 - {}".format(bleu4))
