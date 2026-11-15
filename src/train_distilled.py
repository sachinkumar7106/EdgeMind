
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from dataset import train_loader, validation_loader
from teacher import VisionTransformerTeacher
from student import CNNStudent
from distillation import DistillationLoss


# -------------------- Configuration --------------------

SEED = 42
EPOCHS = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

TEMPERATURE = 4.0
ALPHA = 0.5

TEACHER_PATH = "models/teacher/best_teacher.pth"
STUDENT_PATH = "models/student/best_distilled_student.pth"

os.makedirs("models/student", exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- Evaluation --------------------

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
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


# -------------------- Main Training --------------------

def main():
    print("Using device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Load the pretrained teacher
    teacher = VisionTransformerTeacher().to(device)

    checkpoint = torch.load(
        TEACHER_PATH,
        map_location=device,
        weights_only=True
    )

    teacher.load_state_dict(checkpoint["model_state_dict"])
    teacher.eval()

    # Freeze teacher parameters
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    print(
        "Teacher validation accuracy:",
        f'{checkpoint["val_accuracy"]:.2f}%'
    )

    # Create a fresh student model
    student = CNNStudent().to(device)

    optimizer = AdamW(
        student.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Classification loss for validation
    ce_criterion = nn.CrossEntropyLoss()

    # Reusable knowledge-distillation loss
    distillation_criterion = DistillationLoss(
        temperature=TEMPERATURE,
        alpha=ALPHA
    )

    best_val_accuracy = 0.0

    # -------------------- Epoch Loop --------------------

    for epoch in range(EPOCHS):
        start_time = time.time()

        student.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_index, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Teacher provides targets; no teacher gradients needed
            with torch.no_grad():
                teacher_logits = teacher(images)

            # Student predictions
            student_logits = student(images)

            # Combined CE + knowledge-distillation loss
            loss = distillation_criterion(
                student_logits,
                teacher_logits,
                labels
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (
                student_logits.argmax(dim=1) == labels
            ).sum().item()
            total += labels.size(0)

            if (batch_index + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}] "
                    f"Batch [{batch_index + 1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        train_loss = running_loss / total
        train_accuracy = 100.0 * correct / total

        # Evaluate using true labels, not the distillation loss
        val_loss, val_accuracy = evaluate(
            student,
            validation_loader,
            ce_criterion
        )

        elapsed = time.time() - start_time

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Train accuracy: {train_accuracy:.2f}%")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation accuracy: {val_accuracy:.2f}%")
        print(f"Time: {elapsed / 60:.2f} minutes")

        # Save the best validation checkpoint
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": student.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_accuracy,
                    "temperature": TEMPERATURE,
                    "alpha": ALPHA,
                    "seed": SEED,
                },
                STUDENT_PATH
            )

            print("Best distilled student checkpoint saved!")

        print("-" * 50)

    print("\nDistillation training completed.")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print("Checkpoint:", STUDENT_PATH)


if __name__ == "__main__":
    main()
