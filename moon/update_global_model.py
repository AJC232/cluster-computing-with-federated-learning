import torch

def update_global_model(global_model, local_models, data_sizes):
    # Get the weights from the first model
    avg_weights = local_models[0].state_dict()
    # Calculate total size of all datasets
    total_size = sum(data_sizes)

    # For each weight
    for key in avg_weights.keys():
        # Reset the weight to zero
        avg_weights[key] = 0.0
        # For each model
        for model, data_size in zip(local_models, data_sizes):
            # Add the weighted weights
            avg_weights[key] += (data_size / total_size) * model.state_dict()[key]
    
    # Update the global model
    global_model.load_state_dict(avg_weights)
    
    return global_model

# def update_global_model(global_model, local_models, datasets):
#     total_samples = sum(len(dataset) for dataset in datasets.values())  # Total number of samples across all clients
#     for global_param, *local_params in zip(global_model.parameters(), *(model.parameters() for model in local_models)):
#         weighted_params = torch.stack([local_param.data * (len(dataset) / total_samples) for local_param, dataset in zip(local_params, datasets.values())])
#         global_param.data = torch.sum(weighted_params, dim=0)

#     return global_model
