import sys, os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import time

# add project root so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.resnet import ResNet18   # or whichever model you trained

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================== FUNCTIONS ==================
def load_model(model_path):
    model = ResNet18().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    return transform(pil_img).unsqueeze(0).to(DEVICE)

def inference(model, input_tensor):
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
    return pred_class, probs[0][pred_class].item()

def local_update(model, input_tensor, true_label, client_id, save_dir="outputs"):
    model.train()
    target = torch.tensor([true_label]).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    last_loss = None
    for epoch in range(2):  # fine-tune for 2 epochs
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
        print(f"[Local Update] Epoch {epoch+1}, Loss: {last_loss:.4f}")

    # Unique name with timestamp and client Id
    os.makedirs(save_dir, exist_ok=True)
    filename = f"client{client_id}_update_{int(time.time())}.pth"
    save_path = os.path.join(save_dir, filename)

    torch.save(model.state_dict(), save_path)
    print(f"[Client {client_id}] Local update complete ✅ ({save_path})")

    return last_loss



# ================== MAIN ==================
if __name__ == "__main__":
    img_path = input("Enter path to image: ").strip()
    client_id = input("Enter client ID: ").strip()
    model_path = input("Enter path to model [default: outputs/resnet_global.pth]: ").strip()
    if model_path == "":
        model_path = "outputs/resnet_global.pth"

    # Load model & preprocess image
    model = load_model(model_path)
    input_tensor = preprocess_image(img_path)

    # Inference
    pred_class, conf = inference(model, input_tensor)
    print(f"\nPredicted: {pred_class} | Confidence: {conf*100:.2f}%")

    # Doctor feedback
    feedback = input("Is this prediction correct? (y/n): ").strip().lower()

    if feedback == "y":
        true_label = pred_class
        print("Using prediction as ground truth ✅")
    else:
        true_label = input("Enter correct label (0=benign, 1=malignant): ").strip()
        if true_label not in ["0", "1"]:
            raise ValueError("Invalid label, must be 0 or 1")
        true_label = int(true_label)

    # Local update
    local_update(model, input_tensor, true_label)
