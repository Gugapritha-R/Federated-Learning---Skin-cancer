from utils.data_utils import show_image, data_loader
from utils.federated import Client,federated_avg,evaluate

from models.vgg import VGG
from models.resnet import ResNet18
from models.efficientnet import EfficientNetB3
import torch

if __name__=="__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Show one image
    image_path="dataset/train/benign/3.jpg"
    show_image(image_path)

    # Load data
    train_loader, test_loader, classes = data_loader()
    print("Classes:", classes)  

    # Global FL settings
    num_clients = 5
    num_selected = 2
    num_rounds = 3
    local_epochs = 2

    '''# Simulate splitting dataset among clients (for now all share same loader)
    clients = [Client(i, train_loader) for i in range(num_clients)]
    print("FL Setup:")
    print(f"Num Clients: {num_clients}, Num Selected: {num_selected}, Rounds: {num_rounds}")
    print("Clients:", clients)'''

    #Initializing global model

    global_model=ResNet18().to(device)

    #simulating clients
    clients = [Client(i, train_loader) for i in range(num_clients)]

    # Run federated rounds
    for rnd in range(num_rounds):
        print(f"\n--- Round {rnd+1} ---")

        client_weights = []
        for client in clients[:num_selected]:  # select first N clients (later: randomize)
            optimizer = torch.optim.SGD(global_model.parameters(), lr=0.01)
            w = client.train(global_model, optimizer, epochs=local_epochs)
            client_weights.append(w)

        # Aggregate updates
        new_global_weights = federated_avg(global_model, client_weights)
        global_model.load_state_dict(new_global_weights)

        # Evaluate global model
        acc = evaluate(global_model, test_loader, device)
        print(f"Round {rnd+1} aggregation complete ✅ | Test Accuracy: {acc:.2f}%")

        