
import torch
import torch.nn as nn


class VisionTransformerTeacher(nn.Module):
    def __init__(
        self,
        image_size=32,
        patch_size=4,
        num_classes=10,
        embed_dim=192,
        depth=6,
        num_heads=3
    ):
        super().__init__()

        assert image_size % patch_size == 0

        num_patches = (image_size // patch_size) ** 2

        # Divide the image into patches and project them
        # into embedding vectors.
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Learnable classification token
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        # Input: [batch, 3, 32, 32]
        x = self.patch_embedding(x)

        # [batch, embed_dim, 8, 8]
        x = x.flatten(2).transpose(1, 2)

        # [batch, 64, embed_dim]
        batch_size = x.shape[0]

        cls_tokens = self.cls_token.expand(
            batch_size, -1, -1
        )

        # Add the classification token
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional information
        x = x + self.pos_embedding

        # Learn relationships between image patches
        x = self.transformer(x)

        # Use the classification token
        x = self.norm(x[:, 0])

        # Return raw class scores (logits)
        return self.classifier(x)


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = VisionTransformerTeacher().to(device)

    dummy_images = torch.randn(64, 3, 32, 32).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(dummy_images)

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    print("Teacher model created successfully!")
    print("Device:", device)
    print("Input shape:", dummy_images.shape)
    print("Output shape:", outputs.shape)
    print(f"Parameters: {total_params:,}")
    print("Predicted class IDs:", outputs.argmax(dim=1)[:10])
