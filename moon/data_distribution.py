import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.nn.utils import parameters_to_vector
import torch.nn.functional as F
from torch.utils.data import DataLoader


Num_of_clients = 3

def data_distribution():
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Assuming labels is a list of labels for your dataset
    labels = [i for i in range(10)]

    # Number of clients
    num_clients = 3

    # Create a dictionary to hold the data for each client
    client_data = {i: [] for i in range(num_clients)}

    # For each label in the dataset
    for label in labels:
        # Generate a sample from a Dirichlet distribution
        dirichlet_sample = np.random.dirichlet(np.ones(num_clients)*0.5)

        # Get the indices of the instances in the dataset that have this label
        indices = np.where(np.array(trainset.targets) == label)[0]

        # Shuffle the indices
        np.random.shuffle(indices)

        # Allocate a proportion of the instances of this label to each client
        for i, proportion in enumerate(dirichlet_sample):
            num_samples = int(proportion * len(indices))
            client_data[i].extend(indices[:num_samples])
            indices = indices[num_samples:]

    # Convert the client data to PyTorch datasets
    client_datasets = {i: torch.utils.data.Subset(trainset, indices) for i, indices in client_data.items()}
    
    return client_datasets