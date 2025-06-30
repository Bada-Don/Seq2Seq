# models/decoder.py

import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_token, hidden, cell):
        input_token = input_token.unsqueeze(1)  # (batch, 1)
        embedded = self.embedding(input_token)  # (batch, 1, embed_dim)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))  # (batch, output_dim)
        return prediction, hidden, cell
