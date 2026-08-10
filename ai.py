import copy
import torch
import torch.nn as nn

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def new(input_len: int, output_len: int, hidden_layers: list[int]) -> nn.Sequential:
    """
    Creates a new neural network model.
    
    :param input_len: Number of input features
    :param output_len: Number of output features
    :param hidden_layers: List containing node counts for hidden layers, e.g., [64, 32]
    :return: A PyTorch nn.Sequential model
    """
    layers = []
    prev_dim = input_len
    
    # Add hidden layers with ReLU activations
    for hidden_dim in hidden_layers:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        prev_dim = hidden_dim
        
    # Output layer
    layers.append(nn.Linear(prev_dim, output_len))
    
    model = nn.Sequential(*layers).to(device)
    return model


def mutate(model: nn.Module, mutation_rate: float = 0.1, mutation_power: float = 0.05) -> nn.Module:
    """
    Returns a new deep copy of the model with slightly mutated weights and biases.
    
    :param model: The original model
    :param mutation_rate: Probability of mutating a specific weight (0.0 to 1.0)
    :param mutation_power: Standard deviation of Gaussian noise added to weights
    :return: A new mutated PyTorch model
    """
    # Create a deep copy so the original model remains unchanged
    mutated_model = copy.deepcopy(model)
    
    with torch.no_grad():
        for param in mutated_model.parameters():
            # Create a boolean mask indicating which weights mutate
            mutation_mask = torch.rand_like(param) < mutation_rate
            # Generate random normal noise
            noise = torch.randn_like(param) * mutation_power
            # Apply noise where the mask is True
            param.add_(noise * mutation_mask)
            
    return mutated_model


def run(inputs: list[float] | torch.Tensor, model: nn.Module) -> torch.Tensor:
    """
    Runs an input vector through the model and returns the output tensor.
    
    :param inputs: List or Tensor of input values matching input_len
    :param model: PyTorch model
    :return: PyTorch Tensor containing model outputs
    """
    model.eval()
    with torch.no_grad():
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        
        # Ensure tensor is on the right device and has batch dimension
        inputs = inputs.to(device)
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
            
        output = model(inputs)
        return output.squeeze(0)  # Return single dimension if single input batch


def save(model: nn.Module, file_path: str) -> None:
    """
    Saves the model architecture and state dictionary to disk.
    
    :param model: Model to save
    :param file_path: Path to save file (.pt or .pth)
    """
    torch.save(model, file_path)


def load(file_path: str) -> nn.Module:
    """
    Loads a saved model from disk.
    
    :param file_path: Path to saved model file
    :return: Loaded PyTorch model
    """
    model = torch.load(file_path, weights_only=False)
    model.to(device)
    model.eval()
    return model
