import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.utils import parameters_to_vector
import torch.nn.functional as F
import socket
import pickle
import threading


class BaseEncoder(nn.Module):
    def __init__(self):
        super(BaseEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ProjectionHead(nn.Module):
    def __init__(self):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(84, 256)
        self.fc2 = nn.Linear(256, 128)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x) 
        x = self.fc2(x)
        return x


class MOON(nn.Module):
    def __init__(self):
        super(MOON, self).__init__()
        self.base_encoder = BaseEncoder()
        self.projection_head = ProjectionHead()
        # self.output_layer = nn.Linear(256, 10)
        self.output_layer = nn.Linear(128, 10)


    def forward(self, x):
        x = self.base_encoder(x)
        x = self.projection_head(x)
        x = self.output_layer(x)
        return x

def cosine_similarity(model1, model2):
    vec1 = parameters_to_vector(model1.parameters())
    vec2 = parameters_to_vector(model2.parameters())

    cos_sim = vec1.dot(vec2) / (vec1.norm() * vec2.norm())

    return cos_sim

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='client', help='server or client')
    args = parser.parse_args()
    return args


def update_global_model(global_model, local_models):
    for global_param, *local_params in zip(global_model.parameters(), *(model.parameters() for model in local_models)):
        global_param.data = torch.mean(torch.stack([local_param.data for local_param in local_params]), dim=0)

    return global_model


class Communication:
    def __init__(self, host='10.100.64.46', port=12345):
        self.host = host
        self.port = port


    def init_server(self,):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.host, self.port))
        self.server.listen(1)
        print(f"Server started at {self.host}:{self.port}")


    def init_client(self,):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((self.host, self.port))
        print(f"Connected to server at {self.host}:{self.port}")


    def send_model(self, model, conn):
        data = pickle.dumps(model.state_dict())
        conn.send(data)

    def receive_model(self, conn):
        data = b""
        print("Receiving data...") 
        while True:
            packet = conn.recv(1024)  
            if not packet: 
                break
            data += packet 

        model_state_dict = pickle.loads(data)

        model = MOON()
        model.load_state_dict(model_state_dict)
        print("Data received successfully.") 
        return model

    def server_accept(self):
        conn, addr = self.server.accept()
        print(f"Connection established with {addr}")
        return conn

    def close_connection(self, conn):
        conn.close()

    def close_server(self):
        self.server.close()

    def close_client(self):
        self.client.close()

def R(model, x):
    # Use the output of the projection head as the representation
    with torch.no_grad():
        x = model.base_encoder(x)
        x = model.projection_head(x)
    return x

def sim(z1, z2):
    # Use the cosine similarity as the similarity function
    return torch.nn.functional.cosine_similarity(z1, z2, dim=1)

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.supervisedLoss = nn.CrossEntropyLoss()
        
    def forward(self, z, labels, zglob, zprev, tau, mue):
        supervised_loss = self.supervisedLoss(outputs, labels)
        
        numerator = torch.exp(sim(z, zglob) / tau)
        denominator = torch.exp(sim(z, zglob) / tau) + torch.exp(sim(z, zprev) / tau)
        contrastive_loss = -torch.log(numerator / denominator)
        
        loss = supervised_loss + mue * contrastive_loss
        return loss


if __name__ == "__main__":
    args = getArgs()

    if args.type == 'server':
       
        global_model = MOON()
        comm = Communication()
        comm.init_server()


        for round in range(20):
            local_models = []
            for _ in range(2):  # Accept connections from 3 clients
                conn = comm.server_accept()
                print("sending model to client...")
                comm.send_model(global_model, conn)  # Send the global model to the client
                print("model sent to client")
                comm.close_connection(conn)

                print("waiting...")
                conn = comm.server_accept()
                local_model = comm.receive_model(conn)  # Receive the updated model from the client
                print("local model to recieved")
                local_models.append(local_model)
                comm.close_connection(conn)


            global_model = update_global_model(global_model, local_models)

        comm.close_server()

    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

        comm = Communication()

        # Define the hyper-parameter mu
        mu = 0.01

        for round in range(5): 
            comm.init_client()
            # Receive the global model from the server
            global_model = comm.receive_model(comm.client)
            print("global model recieved", global_model)
            comm.close_client()

            #Make a copy of global model
            local_model = global_model

            # Define a loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(local_model.parameters(), lr=0.001, momentum=0.9)

            # Train the model on the CIFAR-10 dataset
            for epoch in range(1):
                print("Epoch", epoch+1, "started...")
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data

                    if i==0:
                        prev_model = local_model

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward + backward + optimize
                    outputs = local_model(inputs)
                    lsup = criterion(outputs, labels)
                    z = R(local_model, inputs)
                    zglob = R(global_model, inputs)
                    zprev = R(prev_model, inputs)
                    lcon = -torch.nn.functional.log_softmax(torch.stack([sim(z, zglob), sim(z, zprev)], dim=1) / 0.1, dim=1)[:, 0].mean()
                    l = lsup + mu * lcon
                    l.backward()
                    optimizer.step()

                    prev_model = local_model

                    # Print statistics
                    running_loss += l.item()
                    if i % 2000 == 1999:  # Print every 2000 mini-batches
                        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                        running_loss = 0.0

            print('Finished Training')

            # Calculate accuracy on the test set
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    outputs = local_model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

            # Send the updated model back to the server
            print("sending model to server...")
            comm.init_client()
            comm.send_model(global_model, comm.client)
            print("model sent to server")
            comm.close_client()


