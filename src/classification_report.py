
import os
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from dataset import test_loader
from student import CNNStudent


MODEL_PATH = "models/student/best_distilled_student.pth"

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # Load the best distilled student
    model = CNNStudent().to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_labels = []
    all_predictions = []

    # Collect predictions for the complete test set
    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu()

            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())

    # Generate classification metrics
    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    print("\nCLASSIFICATION REPORT")
    print("=" * 70)
    print(report)

    report_path = os.path.join(
        OUTPUT_DIR, "classification_report.txt"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("EdgeMind - Distilled CNN Classification Report\n\n")
        file.write(report)

    print("Report saved to:", report_path)

    # Generate confusion matrix
    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(CLASS_NAMES))),
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Number of images")

    ax.set(
        xticks=range(len(CLASS_NAMES)),
        yticks=range(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted class",
        ylabel="Actual class",
        title="EdgeMind: Distilled CNN Confusion Matrix",
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    # Display counts inside the matrix
    threshold = matrix.max() / 2

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if matrix[row, column] > threshold
                    else "black"
                ),
                fontsize=8,
            )

    fig.tight_layout()

    matrix_path = os.path.join(
        OUTPUT_DIR, "confusion_matrix.png"
    )

    fig.savefig(matrix_path, dpi=300, bbox_inches="tight")
    print("Confusion matrix saved to:", matrix_path)

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
