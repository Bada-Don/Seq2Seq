import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

    def forward(self, src, src_lengths):
        embedded = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths, batch_first=True, enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.lstm(packed)
        
        # Unpack sequence
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        
        # Return all encoder outputs, not just the last one
        return outputs, hidden, cell