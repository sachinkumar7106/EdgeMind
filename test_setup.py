import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

    x = torch.rand(5000, 5000).cuda()
    y = torch.rand(5000, 5000).cuda()

    z = torch.matmul(x, y)

    print("GPU computation successful!")
    print("Result device:", z.device)
else:
    print("Running on CPU")