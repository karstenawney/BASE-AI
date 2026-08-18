import ctypes
import numpy as np
import random
import gzip

# Load the compiled shared library
lib = ctypes.CDLL("./neural.so")

# Define C argument types and return type
lib.layer.argtypes = [
    ctypes.POINTER(ctypes.c_int16),  # weights
    ctypes.POINTER(ctypes.c_int16),  # bias
    ctypes.POINTER(ctypes.c_int16),  # input
    ctypes.POINTER(ctypes.c_int16),  # output
    ctypes.c_int,                    # input_length
    ctypes.c_int,                    # output_length
]
lib.layer.restype = None

def layer_c(weights: np.ndarray, bias: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Executes a single layer using the C shared library."""
    input_length = len(x)
    output_length = len(bias)
    output = np.empty(output_length, dtype=np.int16)

    lib.layer(
        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        bias.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        ctypes.c_int(input_length),
        ctypes.c_int(output_length),
    )
    return output

import random

def new(input_len: int, output_len: int, hidden_layers: list[int]) -> list[list[int]]:
    layer_sizes = [input_len] + hidden_layers + [output_len]
    model: list[list[int]] = []
    for i in range(len(layer_sizes) - 1):
        in_dim = layer_sizes[i]
        out_dim = layer_sizes[i + 1]
        num_weights = out_dim * in_dim
        weights = [random.randint(-16384, 16384) for _ in range(num_weights)]
        model.append(weights)
    return model


import random

def mutate(model: list[list[int]], mutation_rate: float = 0.1, mutation_power: float = 0.05) -> list[list[int]]:
    max_delta = int(16384 * mutation_power)
    mutated_model: list[list[int]] = []

    for layer in model:
        mutated_layer: list[int] = []
        for weight in layer:
            if random.random() < mutation_rate:
                # Apply random delta within [-max_delta, max_delta]
                delta = random.randint(-max_delta, max_delta)
                new_weight = weight + delta

                # Clamp to 16-bit signed integer limits
                new_weight = max(-32768, min(32767, new_weight))
                mutated_layer.append(new_weight)
            else:
                mutated_layer.append(weight)

        mutated_model.append(mutated_layer)

    return mutated_model


def run(inputs: list[int], model: list[list[int]]) -> list[int]:
    # Convert initial input to contiguous int16 array
    current_signal = np.array(inputs, dtype=np.int16)

    for weights_flat in model:
        weights_arr = np.array(weights_flat, dtype=np.int16)
        
        # Determine output length based on weight array size vs input size
        input_len = len(current_signal)
        output_len = len(weights_flat) // input_len

        # Dummy bias array since bias is unused in the C function
        bias_arr = np.zeros(output_len, dtype=np.int16)

        # Run layer and propagate signal forward
        current_signal = layer_c(weights_arr, bias_arr, current_signal)

    return current_signal.tolist()


def save(model: list[list[int]], file_path: str) -> None:
    npmodel = np.array(model)
    with gzip.open(file_path, 'wb') as f:
        np.save(f, npmodel)


def load(file_path: str) -> list[list[int]]:
    try:
        with gzip.open(file_path, 'rb') as f:
            model = np.load(f)
    except (IOError, OSError) as e:
        raise ValueError(f"Failed to load model from {file_path}. The file may be corrupted or not a valid PyTorch model.") from e
    return model
