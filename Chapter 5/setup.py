import torch
import accelerate
import bitsandbytes

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Accelerate:", accelerate.__version__)
print("bitsandbytes:", bitsandbytes.__version__)