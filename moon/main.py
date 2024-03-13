import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

from get_args import getArgs
from data_distribution import data_distribution
from communication import Communication
from update_global_model import update_global_model
from loss import Loss
from model import MOON
import threading

Rounds = 20
Epochs = 10
Num_of_clients = 3

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
    
    datasets = data_distribution()      
    
    if args.type == 'server':
        start = time.time()
       
        global_model = MOON()
            
        comm = Communication()
        comm.init_server()

        for round in range(Rounds):
            local_models = []
            threads = []
            for _ in range(Num_of_clients):  
                # conn = comm.server_accept()
                # print("sending model to client...")
                # comm.send_model(global_model, conn)  
                # print("model sent to client")

                # print("waiting...")
                # local_model = comm.receive_model(conn)  
                # print("local model to recieved")
                # local_models.append(local_model)
                # print("--------", len(local_models))
                
                # comm.close_connection(conn)
                
                conn = comm.server_accept()
                thread = threading.Thread(target=process_client, args=(conn, global_model, local_models))
                thread.start()
                threads.append(thread)

            for thread in threads:
                thread.join()

            global_model = update_global_model(global_model, local_models, datasets)

        comm.close_server()
        
        global_model.eval()

        # Load the CIFAR-10 test dataset
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = global_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy of the global model on the CIFAR-10 test images: {accuracy} %')
        
        end = time.time()
        total_time = (end - start)/60
        print(f"Total time: {total_time} minutes")

    else:
        start = time.time()
        
        # # Load the CIFAR-10 dataset
        # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(datasets[args.client_id], batch_size=4, shuffle=True, num_workers=2)

        # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

        comm = Communication()

        for round in range(Rounds): 
            print("Round", round+1, "started...")
            comm.init_client()
            # Receive the global model from the server
            global_model = comm.receive_model(comm.client)
            print("global model recieved")
            # comm.close_client()
            
            #Make a copy of global model
            local_model = global_model

            # Define a loss function and optimizer
            criterion = Loss()
            # criterion = CrossEntropyLoss()
            optimizer = optim.SGD(local_model.parameters(), lr=0.001, momentum=0.9)

            # Train the model on the CIFAR-10 dataset
            for epoch in range(Epochs):
                print("Epoch", epoch+1, "started...")
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data

                    if i==0:
                        prev_model = torch.zeros_like(local_model(inputs))

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward + backward + optimize
                    outputs = local_model(inputs)
                    global_outputs = global_model(inputs)
                    if i == 0:
                        loss = criterion(outputs, labels, global_outputs, prev_model, 1, 0.5)
                    else:
                        prev_outputs = prev_model(inputs)
                        loss = criterion(outputs, labels, global_outputs, prev_outputs, 1, 0.5)
                        
                    # loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    prev_model = local_model

                    # Print statistics
                    running_loss += loss.item()
                    if i % 2000 == 1999:  # Print every 2000 mini-batches
                        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                        running_loss = 0.0

            print('Finished Training')

            # # Calculate accuracy on the test set
            # correct = 0
            # total = 0
            # with torch.no_grad():
            #     for data in testloader:
            #         images, labels = data
            #         outputs = local_model(images)
            #         _, predicted = torch.max(outputs.data, 1)
            #         total += labels.size(0)
            #         correct += (predicted == labels).sum().item()

            # print('Accuracy of the local model on the test images: %d %%' % (100 * correct / total))

            # Send the updated model back to the server
            print("sending model to server...")
            comm.send_model(local_model, comm.client)
            print("model sent to server")
            comm.close_client()
            
            
        end = time.time()
        total_time = (end - start)/60
        print(f"Total time: {total_time} minutes")
            