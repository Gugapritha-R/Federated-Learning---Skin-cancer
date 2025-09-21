import sys, os
import torch

# add project root so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.resnet import ResNet18   # same model used by clients

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def average_weights(weight_list):
    """ Federated averaging of model weights """
    avg_weights = {}
    for key in weight_list[0].keys():
        avg_weights[key] = sum([weights[key] for weights in weight_list]) / len(weight_list)
    return avg_weights


if __name__ == "__main__":
    # Ask user for local updates
    update_paths = input("Enter local update files (comma separated): ").strip().split(",")

    update_paths = [path.strip() for path in update_paths if path.strip() != ""]
    if len(update_paths) == 0:
        raise ValueError("No update files provided")

    # Load updates
    weight_list = []
    for path in update_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Update file not found: {path}")
        print(f"Loading {path}")
        weights = torch.load(path, map_location=DEVICE)
        weight_list.append(weights)

    # Aggregate updates
    print("\nAggregating updates...")
    new_global_weights = average_weights(weight_list)

    # Load model & apply new weights
    global_model = ResNet18().to(DEVICE)
    global_model.load_state_dict(new_global_weights)

    # Save new global model
    save_path = "outputs/global_model.pth"
    torch.save(global_model.state_dict(), save_path)
    print(f"✅ New global model saved to {save_path}")
