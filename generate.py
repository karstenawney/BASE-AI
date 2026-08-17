import math
import random
from ai import load, run

alphabet = "abcdefghijklmnopqrstuvwxyz "

def encode(sequence: list[int]) -> list[float]:
    """Converts a list of integer character indices (0-26) into a flattened one-hot list of length 27 * len(sequence)."""
    encoded = []
    for char_idx in sequence:
        vec = [0.0] * 27
        # Clamp index safely between 0 and 26
        safe_idx = max(0, min(26, int(char_idx)))
        vec[safe_idx] = 1.0
        encoded += vec
    return encoded


def choose_char(hotlist):
    minimum = min(hotlist)
    diff = max(hotlist) - minimum
    if diff > 0:
        hotlistmapped = [(hotlistitem - minimum) / diff for hotlistitem in hotlist]
    else:
        hotlistmapped = [0.0] * len(hotlist)
    for i in range(len(hotlistmapped)):
        print(f"{alphabet[i]}: {hotlistmapped[i]}")
    chosen = hotlist.index(max(hotlist))
    return chosen

def main():
    file_path = input("Enter model path: ")
    model = load(file_path)

    prompt = list(range(10))

    print("\n--- Generating Text (Press Enter for next token) ---\n")

    while True:
        context = prompt[-10:]
        inputs = encode(context)

        # Get logits tensor/list from model
        logits = run(inputs, model)

        predicted_char = choose_char(logits)

        prompt.append(predicted_char)

        text_string = "".join(
            alphabet[idx] if idx < len(alphabet) else "?" for idx in prompt
        )

        print(f"Token Indices: {prompt}")
        print(f"Generated Text: {text_string}\n")

        input()


if __name__ == "__main__":
    main()
