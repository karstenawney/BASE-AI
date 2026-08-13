import math
import random
from ai import load, run


def one_hot_encode(sequence):
    encoded = []
    for char_idx in sequence:
        vec = [0.0] * 27
        safe_idx = max(0, min(26, int(char_idx)))
        vec[safe_idx] = 1.0
        encoded.extend(vec)
    return encoded


def sample_with_temperature(logits, temperature=0.7):
    """Applies softmax with temperature and samples a token index."""
    # Apply temperature scaling to raw logits
    scaled_logits = [x / temperature for x in logits]

    # Compute softmax
    max_logit = max(scaled_logits)
    exps = [math.exp(x - max_logit) for x in scaled_logits]
    sum_exps = sum(exps)
    probs = [e / sum_exps for e in exps]

    # Weighted random choice based on probabilities
    r = random.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r <= cumulative:
            return i
    return len(logits) - 1  # Fallback to last index


def main():
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    file_path = input("Enter model path: ")
    model = load(file_path)

    prompt = [19, 7, 8, 18, 26, 8, 18, 26, 5, 17]

    print("\n--- Generating Text (Press Enter for next token) ---\n")

    while True:
        context = prompt[-10:]
        one_hot_inputs = one_hot_encode(context)

        # Get logits tensor/list from model
        logits = run(one_hot_inputs, model)
        if hasattr(logits, "tolist"):
            logits = logits.tolist()

        # Sample next token with temperature (adjust 0.7 to tweak randomness)
        predicted_char = sample_with_temperature(logits, temperature=0.7)

        prompt.append(predicted_char)

        text_string = "".join(
            alphabet[idx] if idx < len(alphabet) else "?" for idx in prompt
        )

        print(f"Token Indices: {prompt}")
        print(f"Generated Text: {text_string}\n")

        input()


if __name__ == "__main__":
    main()