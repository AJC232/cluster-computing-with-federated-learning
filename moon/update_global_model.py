import torch

def update_global_model(global_model, local_models, datasets):
    total_samples = sum(len(dataset) for dataset in datasets.values())  # Total number of samples across all clients
    for global_param, *local_params in zip(global_model.parameters(), *(model.parameters() for model in local_models)):
        weighted_params = torch.stack([local_param.data * (len(dataset) / total_samples) for local_param, dataset in zip(local_params, datasets.values())])
        global_param.data = torch.sum(weighted_params, dim=0)

    return global_model
