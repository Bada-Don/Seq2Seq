# evaluate.py

import pandas as pd
from infer import infer, load_model
from sklearn.metrics import accuracy_score

def char_level_accuracy(predicted, expected):
    total_chars = 0
    correct = 0
    for pred, exp in zip(predicted, expected):
        length = max(len(pred), len(exp))
        total_chars += length
        correct += sum(p == e for p, e in zip(pred, exp))
    return correct / total_chars if total_chars > 0 else 0.0

def evaluate(test_path):
    df = pd.read_csv(test_path)

    model, input_vocab, target_vocab = load_model()

    total = len(df)
    correct_full = 0
    predicted_list = []
    expected_list = []

    for _, row in df.iterrows():
        eng, expected = row['English'], row['Punjabi']
        predicted, _ = infer(model, input_vocab, target_vocab, eng)

        predicted_list.append(predicted)
        expected_list.append(expected)

        if predicted == expected:
            correct_full += 1

    acc_word = correct_full / total
    acc_char = char_level_accuracy(predicted_list, expected_list)

    print(f"\nResults on {total} test samples:")
    print(f"🟢 Full word accuracy:   {acc_word*100:.2f}%")
    print(f"🟡 Character accuracy:   {acc_char*100:.2f}%")

if __name__ == '__main__':
    evaluate("data/test_data.csv")
