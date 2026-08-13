from ai import run
import random

def example_reward(model):
    # Example reward function here
    # This will define what the AI will do
    # This must return a higher number for good behavior (can be negative)
    # Try to make the reward module's return as stable as possible, maybe implementing monte-carlo simulation
    # Use run(inputs, model)
    return 0


def reward(model):
    scores = []
    for j in range(10):
        inputs = [random.random() for i in range(2)]
        output = run(inputs, model)[0]
        scores.append(abs(output - inputs[0] - inputs[1]))
    return -sum(scores)