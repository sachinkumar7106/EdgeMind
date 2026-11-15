
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


class CNNStudent(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super().__init__()

        # Start from scratch for a fair CIFAR-10 baseline.
        # We can test pretrained weights separately later.
        weights = "DEFAULT" if pretrained else None

        self.model = mobilenet_v3_small(weights=weights)

        # CIFAR-10 has 10 classes instead of ImageNet's 1000.
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(
            in_features, num_classes
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = CNNStudent().to(device)
    model.eval()

    images = torch.randn(64, 3, 32, 32).to(device)

    with torch.no_grad():
        outputs = model(images)

    params = sum(p.numel() for p in model.parameters())

    print("Student model created successfully!")
    print("Device:", device)
    print("Input shape:", images.shape)
    print("Output shape:", outputs.shape)
    print(f"Parameters: {params:,}")
