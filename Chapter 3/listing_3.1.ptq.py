import copy
import torch
import torch.ao.quantization as q

# --- FIX 1: Create a dummy model and dataset ---
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Quantization requires specific "Stubs" at the start and end of the model
        self.quant = torch.ao.quantization.QuantStub()
        self.fc = torch.nn.Linear(10, 5)
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)  # Convert from float to int8
        x = self.fc(x)     # Run the layer
        x = self.dequant(x) # Convert back to float
        return x

# Instantiate the model (this is what your code was missing)
model_fp32 = DummyModel()

# Create a dummy dataset (5 random tensors) for calibration
dataset = [torch.randn(1, 10) for _ in range(5)]
# -----------------------------------------------


# deep copy the original model as quantization is done in place
model_to_quantize = copy.deepcopy(model_fp32)
model_to_quantize.eval()

# get mappings - FIX 2: Corrected typo 'qconfiq' to 'qconfig'
model_to_quantize.qconfig = q.get_default_qconfig("qnnpack")

# prepare
prepared_model = q.prepare(model_to_quantize)

# calibrate - you’ll want to use representative (validation) data.
with torch.inference_mode():
    for x in dataset:
        prepared_model(x)

# quantize - FIX 3: Corrected typo 'prepare_model' to 'prepared_model'
model_quantized = q.convert(prepared_model)

print("Quantization successful!")
print("Original size:", model_fp32.fc.weight.element_size() * model_fp32.fc.weight.nelement(), "bytes")
print("Quantized size:", model_quantized.fc.weight().element_size() * model_quantized.fc.weight().nelement(), "bytes")