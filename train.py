import os
import torch
import torch.nn as nn
from torch.optim import Adam
import pandas as pd

from config import *
from utils.data_utils import get_dataloaders
from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2Seq

from sklearn.model_selection import train_test_split


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_on_val(model, val_loader, criterion):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for src, tgt, src_lens, _ in val_loader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, src_lens, tgt)

            output = output[:, 1:].reshape(-1, output.size(-1))
            tgt = tgt[:, 1:].reshape(-1)

            loss = criterion(output, tgt)
            val_loss += loss.item()

    return val_loss / len(val_loader)

def train():
    # Check if train and test files already exist
    if os.path.exists("data/train_data.csv") and os.path.exists("data/test_data.csv"):
        print("📁 Loading existing train_data.csv and test_data.csv...")
        train_df = pd.read_csv("data/train_data.csv")
        test_df = pd.read_csv("data/test_data.csv")
    else:
        print("🔁 Loading and shuffling data from translit_dataset.csv...")
        full_df = pd.read_csv(DATA_PATH).sample(frac=1).reset_index(drop=True)
        train_df, test_df = train_test_split(full_df, test_size=0.1, random_state=42)

        os.makedirs("data", exist_ok=True)
        train_df.to_csv("data/train_data.csv", index=False)
        test_df.to_csv("data/test_data.csv", index=False)

        print("✅ train_data.csv and test_data.csv saved.")

    train_data = list(zip(train_df['english'], train_df['punjabi']))
    val_data = list(zip(test_df['english'], test_df['punjabi']))

    from utils.data_utils import TransliterationDataset, collate_fn
    from torch.utils.data import DataLoader

    train_dataset = TransliterationDataset(train_data)
    val_dataset = TransliterationDataset(val_data, input_vocab=train_dataset.input_vocab,
                                         target_vocab=train_dataset.target_vocab, build_vocab=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print("⚙️  Building model...")
    encoder = Encoder(len(train_dataset.input_vocab), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    decoder = Decoder(len(train_dataset.target_vocab), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_val_loss = float('inf')

    print(f"🚀 Starting training for {NUM_EPOCHS} epochs...\n")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for src, tgt, src_lens, _ in train_loader:
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()
            output = model(src, src_lens, tgt)

            output = output[:, 1:].reshape(-1, output.size(-1))
            tgt = tgt[:, 1:].reshape(-1)

            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss = evaluate_on_val(model, val_loader, criterion)

        print(f"📅 Epoch [{epoch+1}/{NUM_EPOCHS}] | 🟢 Train Loss: {train_loss:.4f} | 🟡 Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
            torch.save({
                'model': model.state_dict(),
                'input_vocab': train_dataset.input_vocab,
                'target_vocab': train_dataset.target_vocab
            }, CHECKPOINT_PATH)
            print("💾 Best model saved!")

if __name__ == '__main__':
    train()