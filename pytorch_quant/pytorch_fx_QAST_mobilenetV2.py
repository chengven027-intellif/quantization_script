import copy
import torch
import torchvision
from torch.quantization import quantize_fx
import onnx
from tvm import relay

input_fp32 = torch.randn(1, 3, 224, 224)
model_fp = torchvision.models.mobilenet_v2(pretrained=False)

#
# post training dynamic/weight_only quantization
#
model_to_quantize = copy.deepcopy(model_fp)
model_to_quantize.eval()
qconfig_mapping = torch.ao.quantization.QConfigMapping().set_global(
    torch.quantization.default_dynamic_qconfig
)
# prepare
model_prepared = quantize_fx.prepare_fx(
    model_to_quantize, qconfig_mapping, (input_fp32)
)
# no calibration needed when we only have dynamic/weight_only quantization
# quantize
model_quantized = quantize_fx.convert_fx(model_prepared)
y = model_quantized(input_fp32)

#
# post training static quantization
#
model_to_quantize = copy.deepcopy(model_fp)
qconfig_mapping = torch.ao.quantization.QConfigMapping().set_global(
    torch.quantization.get_default_qconfig("qnnpack")
)
model_to_quantize.eval()
# prepare
model_prepared = quantize_fx.prepare_fx(
    model_to_quantize, qconfig_mapping, (input_fp32)
)
# calibrate (not shown)
# quantize
model_quantized = quantize_fx.convert_fx(model_prepared)
y = model_quantized(input_fp32)

#
# quantization aware training for static quantization
#
model_to_quantize = copy.deepcopy(model_fp)
qconfig_mapping = torch.ao.quantization.QConfigMapping().set_global(
    torch.quantization.get_default_qat_qconfig("qnnpack")
)
model_to_quantize.train()
# prepare
model_prepared = quantize_fx.prepare_qat_fx(
    model_to_quantize, qconfig_mapping, (input_fp32)
)
# training loop (not shown)
# quantize
model_quantized = quantize_fx.convert_fx(model_prepared)
y = model_quantized(input_fp32)

scripted_model = torch.jit.trace(model_quantized.eval(), input_fp32)
torch.jit.save(scripted_model, "./data/fx_mode.torchscript")
torch.onnx.export(model_quantized, input_fp32, "./data/fx_mode.onnx", verbose=True)

shape_list = [("input", input_fp32.numpy().shape)]
torchscript_model = torch.jit.load("./data/fx_mode.torchscript")
mod, params = relay.frontend.from_pytorch(torchscript_model, shape_list)
print(mod)

onnx_model = onnx.load("./data/fx_mode.onnx")
mod, params = relay.frontend.from_onnx(
    onnx_model, {"x": input_fp32.numpy().shape}, {"x": "float32"}
)
print(mod)
