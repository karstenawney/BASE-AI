print ("Compiling Helper Functions...")
import sys
import ai
import reward
try:
    from tqdm import tqdm
except (ImportError):
    sys.exit("Please install dependancy: tqdm (pip install tqdm)")
def train(generations: int, population: int, carry: float, startmodel = None, input_len: int = None, output_len: int = None, hidden_layers: list[int] = None):
    """
    Trains a neural network model.

    :param generations: Number of generations of models to go through
    :param population: Number of model variations per generation
    :param carry: Fraction of models that make up the parents of the next generation
    :param startmodel: Optional starting model to begin the training
    :param input_len: Optional Number of input features
    :param output_len: Optional Number of output features
    :param hidden_layers: Optional List containing node counts for hidden layers, e.g., [64, 32]
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

    pbar = tqdm(range(generations))

    for i in pbar:
        models = generation(models, carry)
        pbar.set_description(f"Training... Reward: {reward.reward(models[0])}")

    scores = [reward.reward(m) for m in models]
    best_idx = max(range(len(models)), key=lambda i: scores[i])
    best_model = models[best_idx]
    best_score = scores[best_idx]

    print ("Training Complete")
    print (f"Best Score: {best_score}")
    return best_model

from concurrent.futures import ProcessPoolExecutor

def generation(models: list, carry: float, max_workers: int = None) -> list:
    num_models = len(models)
    carrynum = int(num_models * carry)
    if carrynum == 0:
        raise ValueError(f"Error: carry {carry} is too low for population size {num_models}")

    # Use ThreadPoolExecutor instead of ProcessPoolExecutor if models cannot be pickled
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 1. Parallelize reward calculations
        rewards = list(executor.map(reward.reward, models))
        
        # Sort models based on evaluated rewards
        sorted_models = [
            m for m, _ in sorted(zip(models, rewards), key=lambda pair: pair[1], reverse=True)
        ]
        bestmodels = sorted_models[:carrynum]

        new_per_old = num_models // carrynum
        overflow = num_models % carrynum

        # 2. Collect all input models needing mutation
        models_to_mutate = []
        mutation_counts = []
        for i in range(carrynum):
            count = (new_per_old - 1) + (1 if i < overflow else 0)
            mutation_counts.append(count)
            models_to_mutate.extend([bestmodels[i]] * count)

        # 3. Parallelize mutation calls
        mutated_models = list(executor.map(ai.mutate, models_to_mutate))

    # Reconstruct population matching original order
    newmodels = []
    mut_idx = 0
    for i in range(carrynum):
        newmodels.append(bestmodels[i])
        count = mutation_counts[i]
        newmodels.extend(mutated_models[mut_idx : mut_idx + count])
        mut_idx += count

    return newmodels

def main():
    file = input("Do you have a model file (Y/n): ")
    generations = int(input("Enter number of generations: "))
    population = int(input("Enter number of AI models per generation: "))
    carry = float(input("Enter fraction of models that will survive each generation \n(Ex: 0.1 for 10%): "))
    if file.lower() == "y":
        file = input("Enter model path: ")
        model = ai.load(file)
        model = train(generations, population, carry, startmodel=model)
    else:
        input_len = int(input("Enter number of model inputs: "))
        output_len = int(input("Enter number of model outputs: "))
        hidden_layers_num = int(input("Enter number of hidden layers: "))
        hidden_layers = []
        for i in range(hidden_layers_num):
            hidden_layers.append(int(input(f"Enter size of hidden layer {i + 1}: ")))
        model = train(generations, population, carry, input_len=input_len, output_len=output_len, hidden_layers=hidden_layers)
    file = input("Enter model output path: ")
    ai.save(model, file)

if __name__ == "__main__":
    main()