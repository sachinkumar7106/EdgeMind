
import torch
import torch.nn as nn

from dataset import test_loader
from teacher import VisionTransformerTeacher
from student import CNNStudent


MODELS = {
    "Transformer Teacher": (
        VisionTransformerTeacher,
        "models/teacher/best_teacher.pth",
    ),
    "Baseline CNN Student": (
        CNNStudent,
        "models/student/best_student.pth",
    ),
    "Distilled CNN Student": (
        CNNStudent,
        "models/student/best_distilled_student.pth",
    ),
}


def evaluate_model(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

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

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nEDGE MIND - FINAL TEST EVALUATION")
    print("=" * 55)

    results = {}

    for name, (model_class, checkpoint_path) in MODELS.items():
        print(f"\nEvaluating: {name}")

        model = model_class().to(device)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        test_loss, test_accuracy = evaluate_model(
            model, test_loader, device
        )

        results[name] = test_accuracy

        print(
            f"Best validation accuracy: "
            f'{checkpoint["val_accuracy"]:.2f}%'
        )
        print(f"Test loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_accuracy:.2f}%")

        del model

    print("\n" + "=" * 55)
    print("FINAL COMPARISON")
    print("=" * 55)

    for name, accuracy in results.items():
        print(f"{name}: {accuracy:.2f}%")

    improvement = (
        results["Distilled CNN Student"]
        - results["Baseline CNN Student"]
    )

    print(
        "\nDistillation improvement over baseline: "
        f"{improvement:+.2f} percentage points"
    )


if __name__ == "__main__":
    main()
