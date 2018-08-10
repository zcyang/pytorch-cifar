'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler 

import os
import argparse

from models import *
# from utils import progress_bar
from model_utils import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--output_dir', default="checkpoint", type=str, help='output dir')
parser.add_argument('--model', default="resnet18", type=str, help='model name')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--num_steps', default=5, type=int, help='num steps')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
if args.model == "vgg19":
    net = VGG('VGG19')
elif args.model == "resnet18":
    net = ResNet18()
elif args.model == "resnet50":
    net = ResNet50()
elif args.model == "resnet101":
    net = ResNet101()
# net = PreActResNet18()
# net = GoogLeNet()
elif args.model == "densenet121":
    net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
net = net.to(device)
if device == 'cuda':
    cudnn.benchmark = True

def load_checkpoint():
    checkpoint = torch.load(os.path.join(args.output_dir, 'ckpt.t7'))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    return net

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 350], gamma=0.1)

def finetune():
    net = load_checkpoint()
    features = extract_feature(trainloader, net)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    total = 0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        print("finetune %d "%(batch_idx))
        net = load_checkpoint()
        query = extract_feature(inputs, net)
        top_k_indices = get_nearest_neighbors(features, query, similarity="cos")
        train_batch = trainset[top_k_indices]
        train_inputs, train_targets = train_batch
        train_inputs, train_targets = train_inputs.to("cuda"), train_targets.to("cuda")
        net.train()
        for _ in range(args.num_steps):
          optimizer.zero_grad()
          train_outputs = net(train_inputs)
          loss = criterion(train_outputs, train_targets)
          loss.backward()
          optimizer.step()
          print(loss.items())

        # eval on the single data set
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print("%.3f (%d/%d)"%(100.*correct / total, correct, total))

def test():
    net = load_checkpoint()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


    acc = 100.*correct/total
    print("test acc %.3f"%(acc))


test()
finetune()
