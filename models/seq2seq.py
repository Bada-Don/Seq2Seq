# models/seq2seq.py

import torch
import torch.nn as nn
import random
from config import TEACHER_FORCING_RATIO, MAX_LENGTH

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_lengths, tgt=None, teacher_forcing_ratio=TEACHER_FORCING_RATIO):
        batch_size = src.size(0)
        max_len = tgt.size(1) if tgt is not None else MAX_LENGTH
        tgt_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, max_len, tgt_vocab_size).to(self.device)

        hidden, cell = self.encoder(src, src_lengths)
        input_token = tgt[:, 0] if tgt is not None else torch.full((batch_size,), 1, dtype=torch.long).to(self.device)

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t] = output

            top1 = output.argmax(1)
            input_token = tgt[:, t] if tgt is not None and random.random() < teacher_forcing_ratio else top1

        return outputs
