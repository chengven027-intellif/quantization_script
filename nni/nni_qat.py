import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from nni.compression.pytorch.quantization import QAT_Quantizer
from nni.compression.pytorch.utils.quantization.settings import set_quant_scheme_dtype

# import sys
# sys.path.append('./models')
# from mnist.naive import NaiveModel
from nni_assets.compression.mnist_model import TorchModel, trainer, evaluator

def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('{:2.0f}%  Loss {}'.format(100 * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('Loss: {}  Accuracy: {}%)\n'.format(
        test_loss, 100 * correct / len(test_loader.dataset)))

def prepare_data_loaders(batch_size, data_path):
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

def main():
    # torch.manual_seed(0)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # batch_size = 512
    # data_path = "data/ILSVRC2012/ILSVRC2012_img_val/"
    # train_loader, test_loader = prepare_data_loaders(batch_size, data_path)

    # Two things should be kept in mind when set this configure_list:
    # 1. When deploying model on backend, some layers will be fused into one layer. For example, the consecutive
    # conv + bn + relu layers will be fused into one big layer. If we want to execute the big layer in quantization
    # mode, we should tell the backend the quantization information of the input, output, and the weight tensor of
    # the big layer, which correspond to conv's input, conv's weight and relu's output.
    # 2. Same tensor should be quantized only once. For example, if a tensor is the output of layer A and the input
    # of the layer B, you should configure either {'quant_types': ['output'], 'op_names': ['a']} or
    # {'quant_types': ['input'], 'op_names': ['b']} in the configure_list.

    # configure_list = [{
    #     'quant_types': ['weight', 'input'],
    #     'quant_bits': {'weight': 8, 'input': 8},
    #     'op_names': ['conv1', 'conv2']
    # }, {
    #     'quant_types': ['output'],
    #     'quant_bits': {'output': 8, },
    #     'op_names': ['relu1', 'relu2']
    # }, {
    #     'quant_types': ['output', 'weight', 'input'],
    #     'quant_bits': {'output': 8, 'weight': 8, 'input': 8},
    #     'op_names': ['fc1', 'fc2'],
    # }]

    # you can also set the quantization dtype and scheme layer-wise through configure_list like:
    # configure_list = [{
    #         'quant_types': ['weight', 'input'],
    #         'quant_bits': {'weight': 8, 'input': 8},
    #         'op_names': ['conv1', 'conv2'],
    #         'quant_dtype': 'int',
    #         'quant_scheme': 'per_channel_symmetric'
    #       }]
    # For now quant_dtype's options are 'int' and 'uint. And quant_scheme's options are per_tensor_affine,
    # per_tensor_symmetric, per_channel_affine and per_channel_symmetric.
    # set_quant_scheme_dtype('weight', 'per_channel_symmetric', 'int')
    # set_quant_scheme_dtype('output', 'per_tensor_symmetric', 'int')
    # set_quant_scheme_dtype('input', 'per_tensor_symmetric', 'int')

    # from torchvision import models
    # model = models.resnet50(pretrained=True)
    # dummy_input  = torch.randn(1, 3, 224, 224)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    # # To enable batch normalization folding in the training process, you should
    # # pass dummy_input to the QAT_Quantizer.
    # quantizer = QAT_Quantizer(model, configure_list, optimizer, dummy_input=dummy_input)
    # quantizer.compress()


    config_list = [{
        'quant_types': ['input', 'weight'],
        'quant_bits': {'input': 8, 'weight': 8},
        'op_types': ['Conv2d']
    }, {
        'quant_types': ['output'],
        'quant_bits': {'output': 8},
        'op_types': ['ReLU']
    }, {
        'quant_types': ['input', 'weight'],
        'quant_bits': {'input': 8, 'weight': 8},
        'op_names': ['fc1', 'fc2']
    }]

    model = TorchModel()
    optimizer = torch.optim.SGD(model.parameters(), 1e-2)
    criterion = F.nll_loss


    dummy_input = torch.rand(32, 1, 28, 28)
    quantizer = QAT_Quantizer(model, config_list, optimizer, dummy_input)
    quantizer.compress()
    # print(model)

    for epoch in range(1):
        trainer(model, optimizer, criterion)
        evaluator(model)

    # for epoch in range(40):
    #     print('# Epoch {} #'.format(epoch))
    #     train(model, device, train_loader, optimizer)
    #     test(model, device, test_loader)

    model_path = "TorchModel_nni.pth"
    calibration_path = "TorchModel_nni_calibration.pth"
    onnx_path = "TorchModel_nni.onnx"
    input_shape = (32, 1, 28, 28)
    device = torch.device("cpu")

    calibration_config = quantizer.export_model(model_path, calibration_path, onnx_path, input_shape, device)
    print("Generated calibration config is: ", calibration_config)

if __name__ == '__main__':
    main()   