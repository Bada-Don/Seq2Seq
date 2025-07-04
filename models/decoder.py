import torch
import torch.nn as nn
from .attention import Attention # Import the new module

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.attention = Attention(hidden_dim) # Add the attention layer

        self.embedding = nn.Embedding(output_dim, embed_dim)
        # The LSTM input now includes the context vector from attention
        self.lstm = nn.LSTM(hidden_dim + embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_token, hidden, cell, encoder_outputs):
        # input_token = [batch_size]
        # hidden = [1, batch_size, hidden_dim]
        # cell = [1, batch_size, hidden_dim]
        # encoder_outputs = [batch_size, src_len, hidden_dim]

        input_token = input_token.unsqueeze(1)  # (batch_size, 1)
        embedded = self.embedding(input_token)  # (batch_size, 1, embed_dim)

        # Calculate attention weights
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)
        # a = [batch_size, 1, src_len]

        # Get the weighted context vector
        context = torch.bmm(a, encoder_outputs)
        # context = [batch_size, 1, hidden_dim]

        # Concatenate embedded input token and context vector
        lstm_input = torch.cat((embedded, context), dim=2)
        # lstm_input = [batch_size, 1, embed_dim + hidden_dim]

        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        # prediction = [batch_size, output_dim]

        return prediction, hidden, cell