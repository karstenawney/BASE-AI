import ai
from tqdm import tqdm

def reward(model):
    # Insert reward function here
    # This will define what the AI will do
    # This must return a higher number for good behavior (can be negative)
    # Try to make the reward module's return as stable as possible, maybe implementing monte-carlo simulation
    return 0

def train(input_len: int, output_len: int, hidden_layers: list[int], generations: int, population: int, carry: float, startmodel = None):
    """
    Trains a neural network model.
    
    :param input_len: Number of input features
    :param output_len: Number of output features
    :param hidden_layers: List containing node counts for hidden layers, e.g., [64, 32]
    :param generations: Number of generations of models to go through
    :param population: Number of model variations per generation
    :param carry: Fraction of models that make up the parents of the next generation
    :param model: Optional starting model to begin the training
    :return: A PyTorch nn.Sequential model
    """

    # Create starting model list
    models = []

    if startmodel == None:
        for i in range(population):
            models.append(ai.new(input_len, output_len, hidden_layers))
    else:
        for i in range(population):
            models.append(ai.mutate(startmodel))

    for i in tqdm(range(generations), desc="Training model"):
        models = generation(models, carry)

    scores = []

    for i in range(population):
        scores.append(reward(models[i]))

    score, model = max(zip(scores, models))
    print ("Training Complete")
    print (f"Best Score: {score}")
    return model


def generation(models: list, carry):
    num_models = len(models)
    carrynum = int(num_models * carry)
    if carrynum == 0:
        raise ValueError(f"Error: carry {carry} is too low for population size {num_models}")

    scores = []
    for model in models:
        scores.append(reward(model))

    bestmodels = [model for _, model in sorted(zip(scores, models), reverse=True)[:carrynum]]

    new_per_old = num_models // carrynum
    overflow = num_models % carrynum

    newmodels = []

    for i in range(carrynum):
        newmodels.append(bestmodels[i])
        for j in range(new_per_old - 1):
            newmodels.append(ai.mutate(bestmodels[i]))
        if i < overflow:
            newmodels.append(ai.mutate(bestmodels[i]))
    return newmodels