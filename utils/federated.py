import torch
import torch.nn.functional as F
import copy


class Client:
    def __init__(self, client_id, train_loader):
        self.client_id = client_id
        self.train_loader = train_loader
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    
    
    def train(self, model, optimizer, epochs):
        """Train model locally and return updated weights"""
        model.train()
        for _ in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

        return model.state_dict()
    
    def __repr__(self):
        return f"Client {self.client_id}"



def federated_avg(global_model, client_weights):
    """
    Averages weights from multiple clients to update the global model.
    """
    # Deep copy global weights as a starting point
    new_weights = copy.deepcopy(global_model.state_dict())

    # Average each parameter across clients
    for key in new_weights.keys():
        new_weights[key] = torch.stack([weights[key].float() for weights in client_weights], 0).mean(0)

    return new_weights

def evaluate(model, test_loader, device):
    """
    Evaluate global model on test dataset.
    Returns accuracy (%).
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    acc = 100 * correct / total
    return acc


    
    