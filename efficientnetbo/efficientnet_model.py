import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class SignLanguageEfficientNet(nn.Module):
    """
    Advanced model for Sign Language recognition using EfficientNetB0 backbone.
    This class is configured to handle complex feature extraction.
    """
    def __init__(self, num_classes=26, pretrained=True):
        super(SignLanguageEfficientNet, self).__init__()
        # Load the pre-trained EfficientNetB0
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        
        # Replace the classifier head with a custom one for 26 classes
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass of the model.
        Expects a batch of images with shape (B, C, H, W).
        """
        # Complex feature extraction happens inside the backbone
        return self.backbone(x)

def get_complex_model():
    # Helper to instantiate the model for external scripts
    model = SignLanguageEfficientNet(num_classes=26, pretrained=True)
    return model

if __name__ == "__main__":
    # Quick sanity check
    dummy_model = get_complex_model()
    dummy_input = torch.randn(1, 3, 224, 224)
    print("EfficientNetB0 feature extractor initialized.")
    output = dummy_model(dummy_input)
    print(f"Output shape expected (1, 26), got: {output.shape}")
