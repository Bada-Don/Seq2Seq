# infer.py

import torch
from config import *
from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
from utils.vocab import CharVocab
from anmol_transliterate import transliterate_punjabi
import torch.nn.functional as F
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    torch.serialization.add_safe_globals([CharVocab])

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    input_vocab = checkpoint['input_vocab']
    target_vocab = checkpoint['target_vocab']

    encoder = Encoder(len(input_vocab), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    decoder = Decoder(len(target_vocab), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)

    model = Seq2Seq(encoder, decoder, device).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    return model, input_vocab, target_vocab

def infer(model, input_vocab, target_vocab, text):
    encoded = input_vocab.encode(text)
    src_tensor = torch.tensor(encoded).unsqueeze(0).to(device)
    src_len = [len(encoded)]

    outputs = model(src_tensor, src_len, tgt=None, teacher_forcing_ratio=0.0)
    predictions = outputs.argmax(2).squeeze(0).tolist()

    # Decode Punjabi characters
    punjabi_output = target_vocab.decode(predictions)

    # Convert to AnmolLipi
    anmol_output = ''.join(transliterate_punjabi(list(punjabi_output)))
    return punjabi_output, anmol_output

if __name__ == '__main__':
    model, input_vocab, target_vocab = load_model()
    print("Model loaded.")

    while True:
        text = input("Enter English name (or 'q' to quit): ").strip()
        if text.lower() == 'q':
            break
        punjabi_out, anmol_out = infer(model, input_vocab, target_vocab, text)
        print(f"Punjabi Output: {punjabi_out}")
        print(f"AnmolLipi Output: {anmol_out}")
