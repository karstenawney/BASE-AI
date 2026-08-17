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
        pbar.set_description(f"Training... Loss: {reward.reward(models[0])}")

    scores = []

    for i in range(population):
        scores.append(reward.reward(models[i]))

    best_model = max(models, key=lambda m: reward.reward(m))
    best_score = reward.reward(best_model)
    
    print ("Training Complete")
    print (f"Best Score: {best_score}")
    return best_model

def generation(models: list, carry):
    num_models = len(models)
    carrynum = int(num_models * carry)
    if carrynum == 0:
        raise ValueError(f"Error: carry {carry} is too low for population size {num_models}")

    sorted_models = sorted(models, key=lambda m: reward.reward(m), reverse=True)
    bestmodels = sorted_models[:carrynum]

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