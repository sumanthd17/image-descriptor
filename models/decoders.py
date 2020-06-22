import torch
import torch.nn as nn
from .embedding import WordAndPositionalEmbedding


class TextualHead(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_layers,
        attention_heads,
        feedforward_size,
        dropout,
        max_caption_length,
        padding_idx,
    ):
        super(TextualHead, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention_heads = attention_heads
        self.feedforward_size = feedforward_size
        self.dropout = dropout
        self.padding_idx = padding_idx

        self.embedding = WordAndPositionalEmbedding(
            self.vocab_size,
            self.hidden_size,
            dropout=dropout,
            max_caption_length=max_caption_length,
            padding_idx=padding_idx,
        )

        LayerClass = nn.TransformerDecoderLayer
        _layer = LayerClass(
            self.hidden_size,
            self.attention_heads,
            dim_feedforward=self.feedforward_size,
            dropout=dropout,
            activation="gelu",
        )

        self.decoder = nn.TransformerDecoder(_layer, self.num_layers)
        self.apply(self._init_weights)

        self.output = nn.Linear(self.hidden_size, self.vocab_size)
        self.output.weight = self.embedding.words.weight

    def forward(self, visual_features, caption_tokens, caption_lengths):
        batch_size, max_caption_length = caption_tokens.size()

        ones = torch.ones_like(caption_tokens)
        caption_mask = caption_lengths.unsqueeze(1) < ones.cumsum(dim=1)

        # caption_mask = caption_mask.transpose(0, 1)

        caption_embeddings = self.embedding(caption_tokens)

        unidirectional_mask = self._generate_future_mask(
            max_caption_length, caption_embeddings.dtype, caption_embeddings.device
        )

        caption_embeddings = caption_embeddings.transpose(0, 1)
        # visual_features = visual_features.transpose(0, 1)

        # print(f"vision features: {visual_features.shape}")
        # print(f"caption embed: {caption_embeddings.shape}")
        # print(f"unidirectional mask: {unidirectional_mask.shape}")
        # print(f"key padding mask: {caption_mask.shape}")

        textual_features = self.decoder(
            caption_embeddings,
            visual_features,
            tgt_mask=unidirectional_mask,
            tgt_key_padding_mask=caption_mask,
        )

        textual_features = textual_features.transpose(0, 1)

        output_logits = self.output(textual_features)
        return output_logits

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0, std=0.02)
        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _generate_future_mask(self, size, dtype, device):
        mask = torch.triu(
            torch.ones(size, size, device=device, dtype=dtype), diagonal=1
        )
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask
