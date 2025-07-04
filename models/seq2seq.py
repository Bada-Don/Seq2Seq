import torch
import torch.nn as nn
import random
from config import TEACHER_FORCING_RATIO, MAX_LENGTH, HIDDEN_SIZE # Make sure HIDDEN_SIZE is imported

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        # --- ADD THIS LINE ---
        # This layer will learn the best way to initialize the decoder's hidden state
        # from the encoder's final hidden state.
        self.fc_init_hidden = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        # --------------------

    def forward(self, src, src_lengths, tgt=None, teacher_forcing_ratio=TEACHER_FORCING_RATIO):
        batch_size = src.size(0)
        max_len = tgt.size(1) if tgt is not None else MAX_LENGTH
        tgt_vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, max_len, tgt_vocab_size).to(self.device)

        # encoder_outputs is [batch_size, src_len, hidden_dim]
        # encoder_hidden is [1, batch_size, hidden_dim]
        encoder_outputs, encoder_hidden, encoder_cell = self.encoder(src, src_lengths)

        # --- USE THE NEW LAYER TO INITIALIZE DECODER STATE ---
        # Pass the encoder's final hidden state through the linear layer
        # and use tanh for activation. This is the decoder's starting point.
        hidden = torch.tanh(self.fc_init_hidden(encoder_hidden))
        
        # We can initialize the cell state similarly or just use the encoder's cell state.
        # Let's use the encoder's cell state for simplicity for now.
        cell = encoder_cell
        # --------------------------------------------------------

        input_token = tgt[:, 0] if tgt is not None else torch.full((batch_size,), 1, dtype=torch.long).to(self.device)

        for t in range(1, max_len):
            # Pass encoder_outputs to the decoder in each step
            output, hidden, cell = self.decoder(input_token, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input_token = tgt[:, t] if tgt is not None and random.random() < teacher_forcing_ratio else top1

        return outputs