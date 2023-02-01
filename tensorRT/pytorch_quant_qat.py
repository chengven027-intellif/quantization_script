import torch
import torchvision
import torch.utils.data
from torch import nn

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

from torchvision import models
import torchvision.transforms as transforms

import os

import time
from tqdm import tqdm, trange

import onnx
from tvm import relay

def prepare_data_loaders(data_path):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = torchvision.datasets.ImageFolder(
        data_path, transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    dataset_test = torchvision.datasets.ImageFolder(
        data_path, transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        sampler=train_sampler)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size,
        sampler=test_sampler)

    return data_loader, data_loader_test


def collect_stats(model, data_loader, num_batches):
     """Feed data to the network and collect statistic"""

     # Enable calibrators
     for name, module in model.named_modules():
         if isinstance(module, quant_nn.TensorQuantizer):
             if module._calibrator is not None:
                 module.disable_quant()
                 module.enable_calib()
             else:
                 module.disable()

     for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        if use_cuda:
            model(image.cuda())
        else:
            model(image)
        if i >= num_batches:
            break

     # Disable calibrators
     for name, module in model.named_modules():
         if isinstance(module, quant_nn.TensorQuantizer):
             if module._calibrator is not None:
                 module.enable_quant()
                 module.disable_calib()
             else:
                 module.enable()

def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")
    if use_cuda:
        model.cuda()


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

def evaluate(model, criterion, data_loader, neval_batches):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print('.', end = '')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt >= neval_batches:
                 return top1, top5

    return top1, top5



def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
    model.train()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')

    cnt = 0
    for image, target in data_loader:
        start_time = time.time()
        print('.', end = '')
        cnt += 1
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        avgloss.update(loss, image.size(0))
        if cnt >= ntrain_batches:
            print('Loss', avgloss.avg)

            print('Training: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
            return

    # print('Full imagenet train set:  * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
    #       .format(top1=top1, top5=top5))
    return



if __name__ == "__main__":
    data_path = "data/ILSVRC2012/ILSVRC2012_img_val/"
    saved_model_dir = "./data/"
    batch_size = 512

    data_loader, data_loader_test = prepare_data_loaders(data_path)

    # sys.path.append("path to torchvision/references/classification/")
    # from train import evaluate, train_one_epoch, load_data

    # adding quantized modules
    from pytorch_quantization import quant_modules
    quant_modules.initialize()

    # Cuda 
    use_cuda = 0

    # calibration
    quant_desc_input = QuantDescriptor(calib_method='histogram')
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

    model = models.resnet50()

    if use_cuda:
        model.cuda()

    # It is a bit slow since we collect histograms on CPU
    with torch.no_grad():
        collect_stats(model, data_loader, num_batches=2)
        compute_amax(model, method="percentile", percentile=99.99)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Training takes about one and half hour per epoch on a single V100
    train_one_epoch(model, criterion, optimizer, data_loader, "cpu", 100)

    # Save the model
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    example_inputs = torch.randn(1, 3, 224, 224)
    trt_model_file_path_torch = saved_model_dir + "MobileNetV2_trt_quantized_qat.pth"
    trt_model_file_path_onnx = saved_model_dir + "MobileNetV2_trt_quantized_qat.onnx"

    scripted_model = torch.jit.trace(model.eval(), example_inputs)
    torch.jit.save(scripted_model, trt_model_file_path_torch)
    torch.onnx.export(
        model, example_inputs, trt_model_file_path_onnx, verbose=False
    )

    # shape_list = [("input", example_inputs.numpy().shape)]
    # torchscript_model = torch.jit.load(trt_model_file_path_torch)
    # mod, params = relay.frontend.from_pytorch(torchscript_model, shape_list)
    # mod = relay.transform.InferType()(mod)
    # print(mod)
    # print("*************Print Pytorch IR Success************************")

    onnx_model = onnx.load(trt_model_file_path_onnx)
    mod, params = relay.frontend.from_onnx(
        onnx_model, {"x": example_inputs.numpy().shape}, {"x": "float32"}
    )
    mod = relay.transform.InferType()(mod)
    print(mod)
    print("*************Print Onnx IR Success************************")

    # example_inputs = torch.randn(1, 3, 224, 224)
    # onnx_model = onnx.load("./data/MobileNetV2_trt_quantized_qat.onnx")
    # mod, params = relay.frontend.from_onnx(
    #     onnx_model, {"inputs.1": example_inputs.numpy().shape}
    # )
    # mod = relay.transform.InferType()(mod)
    # print(mod)
    # print("*************Print Onnx IR Success************************")

    