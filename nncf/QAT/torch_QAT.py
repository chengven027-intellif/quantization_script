import sys
import time
import warnings  # To disable warnings on export to ONNX.
import zipfile
from pathlib import Path
import logging

import torch
import nncf  # Important - should be imported directly after torch.

import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from nncf.common.utils.logger import set_log_level
set_log_level(logging.ERROR)  # Disables all NNCF info and warning messages.
from nncf import NNCFConfig
from nncf.torch import create_compressed_model, register_default_init_args
from openvino.runtime import Core
from torch.jit import TracerWarning

# sys.path.append("../utils")
# from notebook_utils import download_file

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

MODEL_DIR = Path("model")
OUTPUT_DIR = Path("output")
DATA_DIR = Path("data")
BASE_MODEL_NAME = "resnet18"
image_size = 64

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Paths where PyTorch, ONNX and OpenVINO IR models will be stored.
fp32_pth_path = Path(MODEL_DIR / (BASE_MODEL_NAME + "_fp32")).with_suffix(".pth")
fp32_onnx_path = Path(OUTPUT_DIR / (BASE_MODEL_NAME + "_fp32")).with_suffix(".onnx")
int8_onnx_path = Path(OUTPUT_DIR / (BASE_MODEL_NAME + "_int8")).with_suffix(".onnx")

# It is possible to train FP32 model from scratch, but it might be slow. Therefore, the pre-trained weights are downloaded by default.
pretrained_on_tiny_imagenet = True


def download_tiny_imagenet_200(
    data_dir: Path,
    # url="http://cs231n.stanford.edu/tiny-imagenet-200.zip",
    tarname="tiny-imagenet-200.zip",
):
    archive_path = data_dir / tarname
    # download_file(url, directory=data_dir, filename=tarname)
    zip_ref = zipfile.ZipFile(archive_path, "r")
    zip_ref.extractall(path=data_dir)
    zip_ref.close()

def prepare_tiny_imagenet_200(dataset_dir: Path):
    # Format validation set the same way as train set is formatted.
    val_data_dir = dataset_dir/'val'
    val_annotations_file = val_data_dir/'val_annotations.txt'
    with open(val_annotations_file, 'r') as f:
        val_annotation_data = map(lambda line: line.split('\t')[:2], f.readlines())
    val_images_dir = val_data_dir/'images'
    for image_filename, image_label in val_annotation_data:
        from_image_filepath = val_images_dir/image_filename
        to_image_dir = val_data_dir/image_label
        if not to_image_dir.exists():
            to_image_dir.mkdir()
        to_image_filepath = to_image_dir/image_filename
        from_image_filepath.rename(to_image_filepath)
    val_annotations_file.unlink()
    val_images_dir.rmdir()


DATASET_DIR = DATA_DIR/"tiny-imagenet-200"
if not DATASET_DIR.exists():
    download_tiny_imagenet_200(DATA_DIR)
    prepare_tiny_imagenet_200(DATASET_DIR)
    print(f"Successfully downloaded and prepared dataset at: {DATASET_DIR}")

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter("Time", ":3.3f")
    losses = AverageMeter("Loss", ":2.3f")
    top1 = AverageMeter("Acc@1", ":2.2f")
    top5 = AverageMeter("Acc@5", ":2.2f")
    progress = ProgressMeter(
        len(train_loader), [batch_time, losses, top1, top5], prefix="Epoch:[{}]".format(epoch)
    )

    # Switch to train mode.
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        images = images.to(device)
        target = target.to(device)

        # Compute output.
        output = model(images)
        loss = criterion(output, target)

        # Measure accuracy and record loss.
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # Compute gradient and do opt step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time.
        batch_time.update(time.time() - end)
        end = time.time()

        print_frequency = 50
        if i % print_frequency == 0:
            progress.display(i)
    
def validate(val_loader, model, criterion):
    batch_time = AverageMeter("Time", ":3.3f")
    losses = AverageMeter("Loss", ":2.3f")
    top1 = AverageMeter("Acc@1", ":2.2f")
    top5 = AverageMeter("Acc@5", ":2.2f")
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix="Test: ")

    # Switch to evaluate mode.
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # Compute output.
            output = model(images)
            loss = criterion(output, target)

            # Measure accuracy and record loss.
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # Measure elapsed time.
            batch_time.update(time.time() - end)
            end = time.time()

            print_frequency = 10
            if i % print_frequency == 0:
                progress.display(i)

        print(" * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5))
    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
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
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


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

num_classes = 200  # 200 is for Tiny ImageNet, default is 1000 for ImageNet
init_lr = 1e-4
batch_size = 128
epochs = 4

model = models.resnet18(pretrained=not pretrained_on_tiny_imagenet)
# Update the last FC layer for Tiny ImageNet number of classes.
model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
model.to(device)

# Data loading code.
train_dir = DATASET_DIR / "train"
val_dir = DATASET_DIR / "val"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    ),
)
val_dataset = datasets.ImageFolder(
    val_dir,
    transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    ),
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, sampler=None
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
)

# Define loss function (criterion) and optimizer.
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

if pretrained_on_tiny_imagenet:
    #
    # ** WARNING: The `torch.load` functionality uses Python's pickling module that
    # may be used to perform arbitrary code execution during unpickling. Only load data that you
    # trust.
    #
    checkpoint = torch.load(str(fp32_pth_path), map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    acc1_fp32 = checkpoint["acc1"]
else:
    best_acc1 = 0
    # Training loop.
    for epoch in range(0, epochs):
        # Run a single training epoch.
        train(train_loader, model, criterion, optimizer, epoch)

        # Evaluate on validation set.
        acc1 = validate(val_loader, model, criterion)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if is_best:
            checkpoint = {"state_dict": model.state_dict(), "acc1": acc1}
            torch.save(checkpoint, fp32_pth_path)
    acc1_fp32 = best_acc1

print(f"Accuracy of FP32 model: {acc1_fp32:.3f}")

dummy_input = torch.randn(1, 3, image_size, image_size).to(device)

torch.onnx.export(model, dummy_input, fp32_onnx_path)
print(f"FP32 ONNX model was exported to {fp32_onnx_path}.")




nncf_config_dict = {
    "input_info": {"sample_size": [1, 3, image_size, image_size]},
    "log_dir": str(OUTPUT_DIR),  # The log directory for NNCF-specific logging outputs.
    "compression": {
        "algorithm": "quantization",  # Specify the algorithm here.
    },
}
nncf_config = NNCFConfig.from_dict(nncf_config_dict)

nncf_config = register_default_init_args(nncf_config, train_loader)

compression_ctrl, model = create_compressed_model(model, nncf_config)

acc1 = validate(val_loader, model, criterion)
print(f"Accuracy of initialized INT8 model: {acc1:.3f}")

compression_lr = init_lr / 10
optimizer = torch.optim.Adam(model.parameters(), lr=compression_lr)

# Train for one epoch with NNCF.
train(train_loader, model, criterion, optimizer, epoch=0)

# Evaluate on validation set after Quantization-Aware Training (QAT case).
acc1_int8 = validate(val_loader, model, criterion)

print(f"Accuracy of tuned INT8 model: {acc1_int8:.3f}")
print(f"Accuracy drop of tuned INT8 model over pre-trained FP32 model: {acc1_fp32 - acc1_int8:.3f}")

if not int8_onnx_path.exists():
    warnings.filterwarnings("ignore", category=TracerWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    # Export INT8 model to ONNX that is supported by OpenVINO™ Toolkit
    compression_ctrl.export_model(int8_onnx_path)
    print(f"INT8 ONNX model exported to {int8_onnx_path}.")