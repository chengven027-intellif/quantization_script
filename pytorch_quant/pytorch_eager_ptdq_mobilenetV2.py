import torch
import torchvision

import onnx
from tvm import relay

input_fp32 = torch.randn(1, 3, 224, 224)
model_fp = torchvision.models.alexnet(pretrained=False)

# create a quantized model instance
model_int8 = torch.quantization.quantize_dynamic(
    model_fp,  # the original model
    {torch.nn.Linear},  # a set of layers to dynamically quantize
    dtype=torch.qint8)  # the target dtype for quantized weights

input_names = ["actual_input"]
output_names = ["output"]
scripted_model = torch.jit.trace(model_int8.eval(), input_fp32)
torch.jit.save(scripted_model, "./data/eager_mode_ptdq.torchscript")

torch.onnx.export(model_int8, 
                  input_fp32,
                  "./data/eager_mode_ptdq.onnx",
                  verbose=False,
                  input_names=input_names,
                  output_names=output_names,
                  export_params=False,
                  opset_version=13,
                  )

shape_list = [("actual_input", input_fp32.numpy().shape)]
torchscript_model = torch.jit.load("./data/eager_mode_ptdq.torchscript")
mod, params = relay.frontend.from_pytorch(torchscript_model, shape_list)
print(mod)
print("*************Print Pytorch IR Success************************")

# onnx_model = onnx.load("./data/eager_mode_ptdq.onnx")
# mod, params = relay.frontend.from_onnx(
#     onnx_model, {"actual_input": input_fp32.numpy().shape}, {"output": "float32"}
# )
# print(mod)
# print("*************Print Onnx IR Success************************")
