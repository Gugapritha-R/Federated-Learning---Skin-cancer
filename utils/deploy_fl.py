import sys, os, torch, cv2, time
import matplotlib.pyplot as plt
import seaborn as sns

# add project root so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.resnet import ResNet18
from utils.data_utils import data_loader
from utils.federated import evaluate_detailed
from utils.inference import preprocess_image, local_update
from utils.aggregate import average_weights

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def capture_image_from_webcam(save_path="live_capture.jpg"):
    cap = cv2.VideoCapture(0)  # webcam
    if not cap.isOpened():
        raise RuntimeError("❌ Could not open webcam")

    print("📸 Press SPACE to capture, ESC to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Live Capture", frame)

        key = cv2.waitKey(1)
        if key % 256 == 27:  # ESC
            print("❌ Capture cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return None
        elif key % 256 == 32:  # SPACE
            cv2.imwrite(save_path, frame)
            print(f"✅ Image saved to {save_path}")
            cap.release()
            cv2.destroyAllWindows()
            return save_path

if __name__ == "__main__":
    # Load test set for evaluation
    _, test_loader, classes = data_loader()

    # Initialize global model
    global_model_path = "outputs/global_model.pth"
    if not os.path.exists(global_model_path):
        print("⚠ No deployment model found, using research model as base")
        global_model_path = "outputs/resnet_global.pth"

    # Track metrics
    acc_history, loss_history = [], []

    # Buffer for batch updates
    update_buffer = []
    batch_size = 3   # aggregate after N client updates

    print("🚀 Deployment started. Type 'exit' anytime to stop.\n")

    scan_count = 0

    while True:
        # Ask doctor for image
        choice = input("Upload image path or type 'live' for webcam (or 'exit' to quit): ").strip()
        if choice.lower() == "exit":
            break

        if choice.lower() == "live":
            img_path = capture_image_from_webcam(f"live_{int(time.time())}.jpg")
            if img_path is None:
                continue
        else:
            img_path = choice
            if not os.path.exists(img_path):
                print(f"❌ File not found: {img_path}")
                continue

        # Load current global model
        model = ResNet18().to(DEVICE)
        model.load_state_dict(torch.load(global_model_path, map_location=DEVICE))

        # Preprocess
        input_tensor = preprocess_image(img_path)

        # Prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            conf = probs[0][pred_class].item()

        print(f"\nPrediction: {classes[pred_class]} | Confidence: {conf*100:.2f}%")

        # Doctor feedback
        feedback = input("Is this prediction correct? (y/n): ").strip().lower()
        if feedback == "y":
            true_label = pred_class
            print("✅ Using prediction as ground truth")
        else:
            true_label = int(input("Enter correct label (0=benign, 1=malignant): ").strip())

        # Local update
        last_loss = local_update(model, input_tensor, true_label, client_id=scan_count)
        loss_history.append(last_loss)

        # Add client update to buffer
        update_buffer.append(model.state_dict())
        scan_count += 1

        # If enough updates → aggregate
        if len(update_buffer) >= batch_size:
            print(f"\n⚡ Aggregating {len(update_buffer)} client updates...")
            new_global_weights = average_weights(update_buffer)
            torch.save(new_global_weights, "outputs/global_model.pth")
            global_model_path = "outputs/global_model.pth"
            update_buffer.clear()  # reset buffer

            # Evaluate updated global model
            global_model = ResNet18().to(DEVICE)
            global_model.load_state_dict(torch.load(global_model_path, map_location=DEVICE))
            acc, precision, recall, f1, cm = evaluate_detailed(global_model, test_loader, DEVICE, classes)
            acc_history.append(acc)

            print(f"📊 Aggregated Model Test Accuracy: {acc:.2f}%")

    # 📊 Plot metrics summary after exit
    if acc_history:
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(range(1, len(loss_history)+1), loss_history, marker='o', label="Train Loss")
        plt.xlabel("Scans Processed")
        plt.ylabel("Loss")
        plt.title("Training Loss per Update")
        plt.grid(True)
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(range(1, len(acc_history)+1), acc_history, marker='o', color="green", label="Test Accuracy")
        plt.xlabel("Aggregations")
        plt.ylabel("Accuracy (%)")
        plt.title("Test Accuracy Over Aggregations")
        plt.grid(True)
        plt.legend()
        plt.show()

        print("\n=== Deployment Session Summary ===")
        print(f"Final Test Accuracy: {acc_history[-1]:.2f}%")
        print(f"Best Test Accuracy: {max(acc_history):.2f}% at aggregation {acc_history.index(max(acc_history))+1}")
