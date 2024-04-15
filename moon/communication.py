import socket
import pickle
from model import MOON

class Communication:
    def __init__(self, host='10.100.64.63', port=9999):
        self.host = host
        self.port = port

    def init_server(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.host, self.port))
        self.server.listen(1)
        print(f"Server started at {self.host}:{self.port}")

    def init_client(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((self.host, self.port))
        print(f"Connected to server at {self.host}:{self.port}")

    def send_dataset(self, dataset, conn):
        data = pickle.dumps(dataset)
        conn.sendall(data + b"<END_OF_TRANSMISSION>")
    
    def receive_dataset(self, conn):
        data = b""
        print("Receiving dataset...")
        while True:
            packet = conn.recv(1024)
            data += packet
            if b"<END_OF_TRANSMISSION>" in data:
                break

        data = data.rstrip(b"<END_OF_TRANSMISSION>")
        dataset = pickle.loads(data)
        return dataset
        

    def send_model(self, model, conn):
        data = pickle.dumps(model.state_dict())
        conn.sendall(data + b"<END_OF_TRANSMISSION>")

    def receive_model(self, conn):
        data = b""
        print("Receiving data...")
        while True:
            packet = conn.recv(1024)
            data += packet
            if b"<END_OF_TRANSMISSION>" in data:
                break

        data = data.rstrip(b"<END_OF_TRANSMISSION>")
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
