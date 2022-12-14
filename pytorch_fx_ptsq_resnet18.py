import os
import sys
import time
import numpy as np

import torch
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, fuse_fx
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
from torchvision.models.resnet import resnet18
import torchvision.transforms as transforms

import onnx
from tvm import relay

# Set up warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.ao.quantization'
)

# Specify random seed for repeatable results
_ = torch.manual_seed(191009)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# def evaluate(model, criterion, data_loader):
#     model.eval()
#     top1 = AverageMeter('Acc@1', ':6.2f')
#     top5 = AverageMeter('Acc@5', ':6.2f')
#     cnt = 0
#     with torch.no_grad():
#         for image, target in data_loader:
#             output = model(image)
#             loss = criterion(output, target)
#             cnt += 1
#             acc1, acc5 = accuracy(output, target, topk=(1, 5))
#             top1.update(acc1[0], image.size(0))
#             top5.update(acc5[0], image.size(0))
#     print('')

#     return top1, top5

def load_model(model_file):
    model = resnet18(pretrained=False)
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to("cpu")
    return model

def print_size_of_model(model):
    if isinstance(model, torch.jit.RecursiveScriptModule):
        torch.jit.save(model, "temp.p")
    else:
        torch.jit.save(torch.jit.script(model), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p")/1e6)
    os.remove("temp.p")

# def prepare_data_loaders(data_path):
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#     dataset = torchvision.datasets.ImageNet(
#         data_path, split="train", transform=transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize,
#         ]))
#     dataset_test = torchvision.datasets.ImageNet(
#         data_path, split="val", transform=transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             normalize,
#         ]))

#     train_sampler = torch.utils.data.RandomSampler(dataset)
#     test_sampler = torch.utils.data.SequentialSampler(dataset_test)

#     data_loader = torch.utils.data.DataLoader(
#         dataset, batch_size=train_batch_size,
#         sampler=train_sampler)

#     data_loader_test = torch.utils.data.DataLoader(
#         dataset_test, batch_size=eval_batch_size,
#         sampler=test_sampler)

#     return data_loader, data_loader_test

# data_path = '~/.data/imagenet'
saved_model_dir = 'data/'
float_model_file = 'resnet18_pretrained_float.pth'

train_batch_size = 30
eval_batch_size = 50

# data_loader, data_loader_test = prepare_data_loaders(data_path)
criterion = nn.CrossEntropyLoss()
float_model = load_model(saved_model_dir + float_model_file).to("cpu")
float_model.eval()

# deepcopy the model since we need to keep the original model around
import copy
model_to_quantize = copy.deepcopy(float_model)

model_to_quantize.eval()

qconfig = get_default_qconfig("fbgemm")
qconfig_dict = {"": qconfig}

input_fp32 = torch.randn(1, 3, 224, 224)

prepared_model = prepare_fx(model_to_quantize, qconfig_dict, (input_fp32))
print(prepared_model.graph)


# def calibrate(model, data_loader):
#     model.eval()
#     with torch.no_grad():
#         for image, target in data_loader:
#             model(image)
# calibrate(prepared_model, data_loader_test)  #

quantized_model = convert_fx(prepared_model)
print(quantized_model)

print("Size of model before quantization")
print_size_of_model(float_model)
print("Size of model after quantization")
print_size_of_model(quantized_model)
# top1, top5 = evaluate(quantized_model, criterion, data_loader_test)
# print("[before serilaization] Evaluation accuracy on test dataset: %2.2f, %2.2f"%(top1.avg, top5.avg))

# fx_graph_mode_model_file_path = saved_model_dir + "resnet18_fx_graph_mode_quantized.pth"

# # save with script
# torch.jit.save(torch.jit.script(quantized_model), fx_graph_mode_model_file_path)
# loaded_quantized_model = torch.jit.load(fx_graph_mode_model_file_path)

# top1, top5 = evaluate(loaded_quantized_model, criterion, data_loader_test)
# print("[after serialization/deserialization] Evaluation accuracy on test dataset: %2.2f, %2.2f"%(top1.avg, top5.avg))

input_fp32 = torch.randn(1, 3, 224, 224)
scripted_model = torch.jit.trace(quantized_model.eval(), input_fp32)
torch.jit.save(scripted_model, "./data/fx_ptsq_resnet18.torchscript")
torch.onnx.export(quantized_model, input_fp32, "./data/fx_ptsq_resnet18.onnx", verbose=False)

shape_list = [("input", input_fp32.numpy().shape)]
torchscript_model = torch.jit.load("./data/fx_ptsq_resnet18.torchscript")
mod, params = relay.frontend.from_pytorch(torchscript_model, shape_list)
print(mod)
print("*************Print Pytorch IR Success************************")

onnx_model = onnx.load("./data/fx_ptsq_resnet18.onnx")
mod, params = relay.frontend.from_onnx(
    onnx_model, {"x": input_fp32.numpy().shape}, {"x": "float32"}
)
print(mod)
print("*************Print Onnx IR Success************************")
