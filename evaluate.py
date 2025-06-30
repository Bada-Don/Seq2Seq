# evaluate.py

import pandas as pd
from infer import infer, load_model

def char_level_accuracy(predicted, expected):
    total_chars = 0
    correct_chars = 0
    for pred, exp in zip(predicted, expected):
        total_chars += max(len(pred), len(exp))
        correct_chars += sum(p == e for p, e in zip(pred, exp))
    return correct_chars / total_chars if total_chars > 0 else 0.0

def evaluate(test_path="data/test_data.csv"):
    print(f"🔍 Evaluating model on: {test_path}")
    df = pd.read_csv(test_path)

    if 'english' not in df.columns or 'punjabi' not in df.columns:
        raise ValueError("CSV must contain 'english' and 'punjabi' columns.")

    model, input_vocab, target_vocab = load_model()

    total = len(df)
    correct_full = 0
    predicted_list = []
    expected_list = []
    failed_cases = []

    for _, row in df.iterrows():
        eng = row['english']
        expected = row['punjabi']
        predicted, _ = infer(model, input_vocab, target_vocab, eng)

        predicted_list.append(predicted)
        expected_list.append(expected)

        if predicted == expected:
            correct_full += 1
        else:
            failed_cases.append((eng, expected, predicted))

    acc_word = correct_full / total
    acc_char = char_level_accuracy(predicted_list, expected_list)

    print(f"\n📊 Results on {total} test samples:")
    print(f"🟢 Full word accuracy:   {acc_word*100:.2f}%")
    print(f"🟡 Character accuracy:   {acc_char*100:.2f}%")

    if failed_cases:
        print("\n❌ Sample failed cases:")
        for i, (eng, exp, pred) in enumerate(failed_cases[:10]):
            print(f"{i+1}. {eng:15} → expected: {exp:10} | predicted: {pred}")

if __name__ == '__main__':
    evaluate()
