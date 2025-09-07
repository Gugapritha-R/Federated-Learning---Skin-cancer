import torch.nn as nn
import timm

class EfficientNetB3(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(EfficientNetB3, self).__init__()
        self.model = timm.create_model("efficientnet_b3", pretrained=pretrained)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
