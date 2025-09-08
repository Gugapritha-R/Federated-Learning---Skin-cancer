from models.vgg import VGG
from models.resnet import ResNet18
from models.efficientnet import EfficientNetB3
from models.densenet import DenseNet121
from models.mobilenet import MobileNetV2
import torch 

device = "cuda" if torch.cuda.is_available() else "cpu"

def model_choice(MODEL_NAME):
    if MODEL_NAME == "resnet":
        global_model = ResNet18().to(device)
    elif MODEL_NAME == "vgg":
        global_model = VGG().to(device)
    elif MODEL_NAME == "efficientnet":
        global_model = EfficientNetB3().to(device)
    elif MODEL_NAME == "densenet":
        global_model = DenseNet121().to(device)
    elif MODEL_NAME == "mobilenet":
        global_model = MobileNetV2().to(device)
    else:
        raise ValueError("Invalid MODEL_NAME")
    
    return global_model