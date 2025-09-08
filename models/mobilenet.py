import torch.nn as nn
import torchvision.models as models

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(MobileNetV2, self).__init__()
        if pretrained:
            self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        else:
            self.model = models.mobilenet_v2(weights=None)

        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
