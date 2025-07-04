# utils/data_utils.py

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from utils.vocab import CharVocab
from config import DATA_PATH, MAX_LENGTH, BASE_DIR

# Change DATA_PATH to point to the master file if it doesn't already
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "train_data.csv")
TEST_DATA_PATH = os.path.join(BASE_DIR, "data", "test_data.csv")

class TransliterationDataset(Dataset):
    def __init__(self, pairs, input_vocab=None, target_vocab=None, build_vocab=True):
        self.pairs = pairs
        self.input_vocab = input_vocab or CharVocab()
        self.target_vocab = target_vocab or CharVocab()

        if build_vocab:
            eng_words = [eng for eng, _ in self.pairs]
            pun_words = [pun for _, pun in self.pairs]
            self.input_vocab.build_vocab(eng_words)
            self.target_vocab.build_vocab(pun_words)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        eng, pun = self.pairs[idx]
        src = self.input_vocab.encode(eng)
        tgt = self.target_vocab.encode(pun)
        return torch.tensor(src), torch.tensor(tgt)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_lengths = [len(seq) for seq in src_batch]
    tgt_lengths = [len(seq) for seq in tgt_batch]

    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)

    return src_padded, tgt_padded, src_lengths, tgt_lengths

def get_dataloaders(batch_size):  # Removed val_split, it's fixed now
    # Read from the permanent train and test files
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    val_df = pd.read_csv(TEST_DATA_PATH)

    train_data = list(zip(train_df['english'], train_df['punjabi']))
    val_data = list(zip(val_df['english'], val_df['punjabi']))

    train_dataset = TransliterationDataset(train_data)  # This builds the vocab
    val_dataset = TransliterationDataset(val_data,
                                         input_vocab=train_dataset.input_vocab,
                                         target_vocab=train_dataset.target_vocab,
                                         build_vocab=False)  # Correctly re-uses vocab

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, train_dataset.input_vocab, train_dataset.target_vocab