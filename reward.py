import math
import random
from ai import run


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


def difference(vec1: list[float], vec2: list[float]) -> float:
    """Computes difference between two characters."""
    return -sum(abs(num1 - num2) for num1, num2 in zip(vec1, vec2))

def reward(model):
    NUM_SAMPLES = 10
    score = 0.0

    text = [0, 11, 15, 7, 0, 1, 4, 19, 26, 26, 26, 26, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 26]

    for _ in range(NUM_SAMPLES):
        start = random.randint(0, len(text) - 11)
        raw_inputs = text[start : start + 10]
        target_char = text[start + 10]  # Single target integer index

        inputs = encode(raw_inputs)
        logits = run(inputs, model)

        # Pick highest-scoring index from output logits
        predicted_char = logits.index(max(logits))

        # Reward +1.0 for correct prediction, 0.0 for wrong prediction
        if predicted_char == target_char:
            score += 1.0
        if predicted_char == 26 and target_char != 26:
            score -= 1.0
        if predicted_char == 26 and target_char == 26:
            score += 10.0

    return score / NUM_SAMPLES