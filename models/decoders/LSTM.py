import torch
import torch.nn as nn


class TextualHeadLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(TextualHeadLSTM, self).__init__()

        self.embedding_layer = nn.Embedding(vocab_size, embed_size)

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, image_features, captions):
        captions = captions[:, :-1]

        embed = self.embedding_layer(captions)
        embed = torch.cat((image_features.unsqueeze(1), embed), dim=1)

        lstm_outputs, _ = self.lstm(embed)
        out = self.linear(lstm_outputs)

        return out

    # Greedy method
    def sample(self, inputs, states=None, max_len=20):
        output_ids = []
        for _ in range(max_len):
            inputs = inputs.unsqueeze(1)
            hidden, states = self.lstm(inputs, states)
            outputs = self.linear(hidden.squeeze(1))
            prediction = outputs.argmax(1)
            output_ids.append(prediction.item())
            inputs = self.embedding_layer(prediction)

        return output_ids
