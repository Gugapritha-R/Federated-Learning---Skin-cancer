import torch.nn as nn
import torchvision.models as models

class DenseNet121(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(DenseNet121, self).__init__()
        if pretrained:
            self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        else:
            self.model = models.densenet121(weights=None)

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
