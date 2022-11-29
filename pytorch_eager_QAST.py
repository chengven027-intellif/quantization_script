import torch
import onnx
from tvm import relay

# define a floating point model where some layers could benefit from QAT
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.quantization.QuantStub()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.bn = torch.nn.BatchNorm2d(1)
        self.relu = torch.nn.ReLU()
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dequant(x)
        return x

# create a model instance
model_fp32 = M()

# model must be set to eval for fusion to work
model_fp32.eval()

# attach a global qconfig, which contains information about what kind
# of observers to attach. Use 'fbgemm' for server inference and
# 'qnnpack' for mobile inference. Other quantization configurations such
# as selecting symmetric or asymmetric quantization and MinMax or L2Norm
# calibration techniques can be specified here.
model_fp32.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

# fuse the activations to preceding layers, where applicable
# this needs to be done manually depending on the model architecture
model_fp32_fused = torch.quantization.fuse_modules(model_fp32,
    [['conv', 'bn', 'relu']])

# Prepare the model for QAT. This inserts observers and fake_quants in
# the model needs to be set to train for QAT logic to work
# the model that will observe weight and activation tensors during calibration.
model_fp32_prepared = torch.quantization.prepare_qat(model_fp32_fused.train())

# run the training loop (not shown)
model_fp32_prepared.train()

# Convert the observed model to a quantized model. This does several things:
# quantizes the weights, computes and stores the scale and bias value to be
# used with each activation tensor, fuses modules where appropriate,
# and replaces key operators with quantized implementations.
model_fp32_prepared.eval()
model_int8 = torch.quantization.convert(model_fp32_prepared)

input_fp32 = torch.randn(4, 1, 4, 4)
# run the model, relevant calculations will happen in int8
res = model_int8(input_fp32)

input_names = ["actual_input"]
output_names = ["output"]
scripted_model = torch.jit.trace(model_int8.eval(), input_fp32)
torch.jit.save(scripted_model, "./data/eager_QAST.torchscript")
torch.onnx.export(model_int8, 
                  input_fp32,
                  "./data/eager_QAST.onnx",
                  verbose=False,
                  input_names=input_names,
                  output_names=output_names,
                  export_params=False,
                  opset_version=13,
                  )

shape_list = [("actual_input", input_fp32.numpy().shape)]
torchscript_model = torch.jit.load("./data/eager_QAST.torchscript")
mod, params = relay.frontend.from_pytorch(torchscript_model, shape_list)
print(mod)
print("*************Print Pytorch IR Success************************")

onnx_model = onnx.load("./data/eager_QAST.onnx")
mod, params = relay.frontend.from_onnx(
    onnx_model, {"actual_input": input_fp32.numpy().shape}, {"output": "float32"}
)
print(mod)
print("*************Print Onnx IR Success************************")