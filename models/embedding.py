import torch
import torch.nn as nn


class WordAndPositionalEmbedding(nn.Module):
    def __init__(
        self, vocab_size, hidden_size, dropout, max_caption_length, padding_idx
    ):
        super(WordAndPositionalEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx

        self.words = nn.Embedding(vocab_size, hidden_size, padding_idx)
        self.positions = nn.Embedding(max_caption_length, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-8, elementwise_affine=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tokens):
        positional_indices = self._create_positional_tokens(tokens)
        word_embeddings = self.words(tokens)
        positional_embeddings = self.positions(positional_indices)

        embeddings = self.layer_norm(word_embeddings + positional_embeddings)
        embeddings = self.dropout(embeddings)

        token_mask = (tokens != self.padding_idx).unsqueeze(-1)

        embeddings = embeddings * token_mask.type(embeddings.dtype)
        return embeddings

    def _create_positional_tokens(self, tokens):
        batch_size, max_caption_length = tokens.size()

        positions = torch.arange(
            max_caption_length, dtype=tokens.dtype, device=tokens.device
        )
        positions = positions.unsqueeze(0).expand(batch_size, max_caption_length)
        return positions
