import os
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

def data_distribution(num_of_clients):
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # transform = transforms.Compose([transforms.ToTensor()])
    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)


    # Assuming labels is a list of labels for your dataset
    labels = [i for i in range(10)]

    # Create a dictionary to hold the data for each client
    client_data = {i: [] for i in range(num_of_clients)}

    # For each label in the dataset
    for label in labels:
        # Generate a sample from a Dirichlet distribution
        dirichlet_sample = np.random.dirichlet(np.ones(num_of_clients)*0.6)

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
    
    sum = 0
    for i in range(len(client_datasets)):
        sum +=len(client_datasets[i]) 
        print(i, len(client_datasets[i]))

    print(sum)

     # Save each client's dataset to a file
    os.makedirs('./datasets', exist_ok=True)
    for i, dataset in client_datasets.items():
        torch.save(dataset, f'./datasets/client_{i}_dataset.pt')

    return client_datasets

# def data_distribution(num_of_clients):
    
#     transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#     trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
#     labels = [i for i in range(10)]

#     client_data = {i: [] for i in range(num_of_clients)}

#     for label in labels:
#         indices = np.where(np.array(trainset.targets) == label)[0]
#         np.random.shuffle(indices)

#         while len(indices) > 0:
#             dirichlet_sample = np.random.dirichlet(np.ones(num_of_clients)*0.5)
#             for i, proportion in enumerate(dirichlet_sample):
#                 num_samples = int(proportion * len(indices))
#                 print(num_samples)
#                 if num_samples == 0:
#                     break
#                 client_data[i].extend(indices[:num_samples])
#                 indices = indices[num_samples:]
#             if num_samples == 0:
#                 break

#     client_datasets = {i: torch.utils.data.Subset(trainset, indices) for i, indices in client_data.items()}
    
#     sum = 0
#     for i in range(len(client_datasets)):
#         sum +=len(client_datasets[i]) 
#         print(i, len(client_datasets[i]))

#     print(sum)

#     # Save each client's dataset to a file
#     os.makedirs('./datasets', exist_ok=True)
#     for i, dataset in client_datasets.items():
#         torch.save(dataset, f'./datasets/client_{i}_dataset.pt')

#     return client_datasets
