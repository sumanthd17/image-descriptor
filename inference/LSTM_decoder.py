import numpy as np

import torch
import torch.nn.functional as F

# Greedy method
def greedy_search_lstm(inputs, decoder, states=None, max_len=50):
    output_ids = []
    for _ in range(max_len):
        inputs = inputs.unsqueeze(1)
        hidden, states = decoder.lstm(inputs, states)
        outputs = decoder.linear(hidden.squeeze(1))
        prediction = outputs.argmax(1)
        output_ids.append(prediction.item())
        inputs = decoder.embedding_layer(prediction)

    return [output_ids]

def beam_search_lstm(encoder_out, decoder, vocab, device, states=None, max_len=50, beam_size=5):

    k = beam_size
    vocab_size = len(vocab)

    encoder_out = encoder_out.unsqueeze(1)  # (1, 1, encoder_dim)
    encoder_dim = encoder_out.size(2)
    
    encoder_out = encoder_out.expand(k, 1, encoder_dim)  # (k, 1, encoder_dim)
    inputs = encoder_out

    k_prev_words = torch.LongTensor([[vocab(vocab.start_word)]] * k).to(device)  # (k, 1)
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)
    seqs = k_prev_words  # (k, 1)

    complete_seqs = list()
    complete_seqs_scores = list()

    step = 1
    while True:
        if(step != 1):
            inputs = decoder.embedding_layer(k_prev_words)

        hidden, states = decoder.lstm(inputs, states)
        outputs = decoder.linear(hidden.squeeze(1))
        scores = F.log_softmax(outputs, dim=1)
        scores = top_k_scores.expand_as(scores) + scores

        if(step == 1):
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)
        
        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        if(step == 1):
            seqs = next_word_inds.unsqueeze(1)
        else:
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != vocab(vocab.end_word)]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        new_states = []
        new_states.append(states[0][:,incomplete_inds,:])
        new_states.append(states[1][:,incomplete_inds,:])
        states = tuple(new_states)

        # Break if things have been going on too long
        if step > max_len:
            break
        step += 1
        
    return complete_seqs