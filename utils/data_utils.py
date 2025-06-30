# utils/data_utils.py

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from utils.vocab import CharVocab
from config import DATA_PATH, MAX_LENGTH

class TransliterationDataset(Dataset):
    def __init__(self, data_path, input_vocab=None, target_vocab=None):
        df = pd.read_csv(data_path)
        self.pairs = list(zip(df['english'], df['punjabi']))

        self.input_vocab = input_vocab or CharVocab()
        self.target_vocab = target_vocab or CharVocab()

        # Build vocabularies
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

def get_dataloaders(batch_size):
    dataset = TransliterationDataset(DATA_PATH)
    input_vocab = dataset.input_vocab
    target_vocab = dataset.target_vocab

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return loader, input_vocab, target_vocab
