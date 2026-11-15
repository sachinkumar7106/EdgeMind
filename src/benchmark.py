
import os
import time
import torch

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

WARMUP_RUNS = 10
BENCHMARK_RUNS = 100


def benchmark_model(model, device):
    model.eval()
    model = model.to(device)

    # One CIFAR-10 image: batch size 1, 3 channels, 32x32
    sample = torch.randn(1, 3, 32, 32, device=device)

    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            model(sample)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.inference_mode():
            for _ in range(BENCHMARK_RUNS):
                model(sample)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

    return (elapsed / BENCHMARK_RUNS) * 1000


def get_model_weights_size_mb(model):
    total_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in model.state_dict().values()
    )
    return total_bytes / (1024 ** 2)


def main():
    print("EDGEMIND MODEL EFFICIENCY BENCHMARK")
    print("=" * 65)

    for name, (model_class, checkpoint_path) in MODELS.items():
        model = model_class()

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        parameters = sum(
            parameter.numel() for parameter in model.parameters()
        )

        weights_size_mb = get_model_weights_size_mb(model)
        print(f"Model weights size: {weights_size_mb:.2f} MB")

        model_size_mb = os.path.getsize(checkpoint_path) / (1024 ** 2)

        print(f"\nModel: {name}")
        print(f"Parameters: {parameters:,}")
        print(f"Checkpoint size: {model_size_mb:.2f} MB")

        for device in [
            torch.device("cpu"),
            torch.device("cuda") if torch.cuda.is_available() else None,
        ]:
            if device is None:
                continue

            latency_ms = benchmark_model(model, device)

            print(
                f"{device.type.upper()} batch-1 latency: "
                f"{latency_ms:.3f} ms/image"
            )

            model = model.to("cpu")

        del model

    print("\nBenchmark completed.")


if __name__ == "__main__":
    main()
