# utils/vocab.py

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

class CharVocab:
    def __init__(self):
        self.char2idx = {
            PAD_TOKEN: 0,
            SOS_TOKEN: 1,
            EOS_TOKEN: 2,
            UNK_TOKEN: 3,
        }
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}

    def build_vocab(self, words):
        for word in words:
            for char in word:
                if char not in self.char2idx:
                    idx = len(self.char2idx)
                    self.char2idx[char] = idx
                    self.idx2char[idx] = char

    def encode(self, text, add_sos_eos=True):
        tokens = [self.char2idx.get(char, self.char2idx[UNK_TOKEN]) for char in text]
        if add_sos_eos:
            tokens = [self.char2idx[SOS_TOKEN]] + tokens + [self.char2idx[EOS_TOKEN]]
        return tokens

    def decode(self, indices, remove_special=True):
        chars = []
        for idx in indices:
            char = self.idx2char.get(idx, UNK_TOKEN)
            if remove_special and char in {PAD_TOKEN, SOS_TOKEN, EOS_TOKEN}:
                continue
            chars.append(char)
        return ''.join(chars)

    def __len__(self):
        return len(self.char2idx)
