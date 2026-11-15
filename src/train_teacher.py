
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from dataset import train_loader, validation_loader as val_loader
from teacher import VisionTransformerTeacher


# ---------------- Configuration ----------------
SEED = 42
EPOCHS = 10
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.05

CHECKPOINT_DIR = "models/teacher"
CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR, "best_teacher.pth"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Use a small batch first to stay within GPU memory limits.
# Change this to 32 if 64 causes an out-of-memory error.
BATCH_SIZE_NOTE = 64


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def main():
    print("Device:", device)

    model = VisionTransformerTeacher().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_index, (images, labels) in enumerate(
            train_loader
        ):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            total += labels.size(0)

            if (batch_index + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}] "
                    f"Batch [{batch_index + 1}/"
                    f"{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        train_loss = running_loss / total
        train_accuracy = 100.0 * correct / total

        val_loss, val_accuracy = evaluate(
            model, val_loader, criterion
        )

        elapsed = time.time() - start_time

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Train accuracy: {train_accuracy:.2f}%")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation accuracy: {val_accuracy:.2f}%")
        print(f"Time: {elapsed / 60:.2f} minutes")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_accuracy,
                    "seed": SEED,
                },
                CHECKPOINT_PATH
            )

            print("Best teacher checkpoint saved!")

        print("-" * 50)

    print("\nTeacher training completed.")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print("Checkpoint:", CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
