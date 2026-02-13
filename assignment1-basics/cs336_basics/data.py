import torch
import numpy as np


def get_batch(
    dataset: np.ndarray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.
    """
    # allowed start indices: [0, len(dataset) - context_length - 1] inclusive
    # effectively range(len(dataset) - context_length)

    # We need context_length + 1 tokens for each example (x and y)
    # The start index i allows retrieving dataset[i : i + context_length + 1]
    # So i + context_length + 1 <= len(dataset)
    # i <= len(dataset) - context_length - 1
    # max_i = len(dataset) - context_length - 1

    high = len(dataset) - context_length
    ix = np.random.randint(0, high, (batch_size,))

    x_list = []
    y_list = []

    for i in ix:
        chunk = dataset[i : i + context_length + 1]
        x_list.append(chunk[:-1])
        y_list.append(chunk[1:])

    x = torch.tensor(np.stack(x_list), dtype=torch.long)
    y = torch.tensor(np.stack(y_list), dtype=torch.long)

    # Handle device
    if "cuda" in device and not torch.cuda.is_available():
        # This will raise the error expected by the test if on CPU
        # But we need to try/except or just let to(device) fail?
        # torch.to() on invalid device usually raises RuntimeError
        pass

    return x.to(device), y.to(device)
