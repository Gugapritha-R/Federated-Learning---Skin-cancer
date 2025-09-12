from utils.data_utils import show_image, data_loader,split_dataset_for_clients
from utils.federated import Client,federated_avg,evaluate,evaluate_detailed
from utils.model_choice import model_choice
from models.vgg import VGG
from models.resnet import ResNet18
from models.efficientnet import EfficientNetB3
from models.densenet import DenseNet121
from models.mobilenet import MobileNetV2

import torch
import random
import matplotlib.pyplot as plt
import seaborn as sns

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
    num_rounds = 10
    local_epochs = 3

    '''# Simulate splitting dataset among clients (for now all share same loader)
    clients = [Client(i, train_loader) for i in range(num_clients)]
    print("FL Setup:")
    print(f"Num Clients: {num_clients}, Num Selected: {num_selected}, Rounds: {num_rounds}")
    print("Clients:", clients)'''

    #Initializing global model

    MODEL_NAME = "resnet" #[]
    global_model=model_choice(MODEL_NAME) 

    
    #splitting dataset for clients
    client_loaders = split_dataset_for_clients(num_clients=num_clients)
    clients = [Client(i, client_loaders[i]) for i in range(num_clients)]

    # Store metrics
    acc_history = []
    loss_history = []


   # Run federated rounds
    for rnd in range(num_rounds):
        print(f"\n--- Round {rnd+1} ---")

        selected_clients = random.sample(clients, num_selected)
        print("Selected clients:", [c.client_id for c in selected_clients])

        client_weights = []
        round_losses = []

        for client in selected_clients:
            optimizer = torch.optim.SGD(global_model.parameters(), lr=0.01)
            w, loss = client.train(global_model, optimizer, epochs=local_epochs)
            client_weights.append(w)
            round_losses.append(loss)

        # Aggregate updates
        new_global_weights = federated_avg(global_model, client_weights)
        global_model.load_state_dict(new_global_weights)

        # Evaluate global model
        acc, precision, recall, f1, cm = evaluate_detailed(global_model, test_loader, device, classes)
        avg_loss = sum(round_losses) / len(round_losses)

        acc_history.append(acc)
        loss_history.append(avg_loss)

        print(f"Round {rnd+1} complete ✅ | Avg Train Loss: {avg_loss:.4f} | Test Accuracy: {acc:.2f}%")

# 📊 Plot metrics
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(range(1, num_rounds+1), loss_history, marker='o', label="Train Loss")
plt.xlabel("Rounds")
plt.ylabel("Loss")
plt.title("Training Loss per Round")
plt.grid(True)
plt.legend()

plt.subplot(1,2,2)
plt.plot(range(1, num_rounds+1), acc_history, marker='o', color="green", label="Test Accuracy")
plt.xlabel("Rounds")
plt.ylabel("Accuracy (%)")
plt.title("Test Accuracy per Round")
plt.grid(True)
plt.legend()
plt.show()

# 🔎 Final metrics
print("\n=== Final Metrics ===")
print(f"Final Test Accuracy: {acc_history[-1]:.2f}%")
print(f"Best Test Accuracy: {max(acc_history):.2f}% at Round {acc_history.index(max(acc_history))+1}")

# 📊 Final evaluation with detailed metrics


print("\n=== Final Evaluation Metrics ===")
print(f"Accuracy : {acc:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall   : {recall:.2f}%")
print(f"F1-score : {f1:.2f}%")
print("\nConfusion Matrix:")
print(cm)


# Plot Confusion Matrix Heatmap
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix - {MODEL_NAME.upper()}")
plt.show()

# Save final trained global model
model_path = f"outputs/{MODEL_NAME}_global.pth"
torch.save(global_model.state_dict(), model_path)
print(f"✅ Global model saved at {model_path}")
