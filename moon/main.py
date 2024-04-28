import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import os
import copy

from get_args import getArgs
from data_distribution import data_distribution
from communication import Communication
from update_global_model import update_global_model
from loss import Loss
from model import MOON
import threading

Rounds = 100
Epochs = 10
Num_of_clients = 1

def process_client(conn, global_model, local_models):        
    print("sending model to client...")
    comm.send_model(global_model, conn)
    print("model sent to client")

    print("waiting...")
    local_model = comm.receive_model(conn)
    print("local model received")
    local_models.append(local_model)
    print("--------", len(local_models))
    comm.close_connection(conn)
    

if __name__ == "__main__":
    args = getArgs()
        
    if args.type == 'server':
        start = time.time()

        datasets = data_distribution(Num_of_clients)

        data_sizes = [len(dataset) for dataset in datasets.values()]

        global_model = MOON()
            
        comm = Communication()
        comm.init_server()

        for round in range(Rounds):
            local_models = []
            threads = []
            global_model.train()
            for i in range(Num_of_clients):                  
                conn = comm.server_accept()
                thread = threading.Thread(target=process_client, args=(conn, global_model, local_models))
                thread.start()
                threads.append(thread)

            for thread in threads:
                thread.join()

            global_model = update_global_model(global_model, local_models, data_sizes)
            
            global_model.eval()
            
            # Load the CIFAR-10 train dataset
            train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)

            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in train_loader:
                    outputs = global_model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f'Accuracy of the global model on the CIFAR-10 train images: {accuracy} %')


            # Load the CIFAR-10 test dataset
            test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in test_loader:
                    outputs = global_model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f'Accuracy of the global model on the CIFAR-10 test images: {accuracy} %')

        comm.close_server()
        
        torch.save(global_model.state_dict(), f'global_model.pth')
        
        global_model.eval()

        # Load the CIFAR-10 test dataset
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = global_model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Final Accuracy of the global model on the CIFAR-10: {accuracy} %')
        
        end = time.time()
        total_time = (end - start)/60
        print(f"Total time: {total_time} minutes")

    else:
        start = time.time()
        
        comm = Communication()
        
        dataset = torch.load(f'./datasets/client_{args.client_id}_dataset.pt')

        # # Load the CIFAR-10 dataset
        # trainloader = torch.utils.data.DataLoader(datasets[args.client_id], batch_size=4, shuffle=True, num_workers=2)
        # print('Dataset size: ', len(datasets[args.client_id]))
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
        print('Dataset size: ', len(dataset))
        
        prev_model = MOON()

        for round in range(Rounds): 
            print("Round", round+1, "started...")
            comm.init_client(9999)
            # Receive the global model from the server
            global_model = comm.receive_model(comm.client)
            print("global model recieved")
            # comm.close_client()
            
            #Make a copy of global model
            local_model = MOON()
            local_model.load_state_dict(global_model.state_dict())

            # Define a loss function and optimizer
            criterion = Loss()
            # criterion = CrossEntropyLoss()
            optimizer = optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9, weight_decay = 0.00001)

            # Define a transform to normalize the data
            # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            for epoch in range(Epochs):
                print("Epoch", epoch+1, "started...")
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
        
                    optimizer.zero_grad()
   
                    local_outputs = local_model(inputs)
                    local_rep = local_model.Rw(inputs)
                    global_rep = global_model.Rw(inputs)
                    prev_rep = prev_model.Rw(inputs)
                    
                    loss = criterion(local_outputs, labels, local_rep, global_rep, prev_rep, 0.5, 3)

                    # Backpropagation and optimization
                    loss.backward()
                    optimizer.step()

                    # Print loss for this batch
                    running_loss += loss.item()
                
                print('Epoch %d, Loss: %.3f' % (epoch + 1, running_loss))
            
            prev_model.load_state_dict(local_model.state_dict())

            print('Finished Training')
            
            local_model.eval()

            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in trainloader:
                    outputs = global_model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f'Accuracy of the local model on the CIFAR-10 train images: {accuracy} %')

            # Load the CIFAR-10 test dataset
            test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in test_loader:
                    outputs = global_model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f'Accuracy of the local model on the CIFAR-10 test images: {accuracy} %')

            local_model.train()
            # Send the updated model back to the server
            print("sending model to server...")
            comm.send_model(local_model, comm.client)
            print("model sent to server")
            comm.close_client()
            
            
        end = time.time()
        total_time = (end - start)/60
        print(f"Total time: {total_time} minutes")
            