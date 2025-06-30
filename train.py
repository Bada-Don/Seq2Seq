# train.py

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from config import *
from utils.data_utils import get_dataloaders
from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2Seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    print("Loading data...")
    loader, input_vocab, target_vocab = get_dataloaders(BATCH_SIZE)

    encoder = Encoder(len(input_vocab), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    decoder = Decoder(len(target_vocab), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for src, tgt, src_lens, _ in loader:
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()
            output = model(src, src_lens, tgt)

            output = output[:, 1:].reshape(-1, output.size(-1))
            tgt = tgt[:, 1:].reshape(-1)

            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {avg_loss:.4f}")

        # Save model
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        torch.save({
            'model': model.state_dict(),
            'input_vocab': input_vocab,
            'target_vocab': target_vocab
        }, CHECKPOINT_PATH)

if __name__ == '__main__':
    train()
