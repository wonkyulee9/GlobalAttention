import torchvision
import torchvision.transforms as transforms
from global_attention import GlobalAttentionNetwork
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import csv


path = './'
device = 'cuda:3'

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

trainset = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=transform_train)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)


testset = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

net = GlobalAttentionNetwork()
net = net.to(device)
num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(num_params)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, 65000)


def train(epoch, global_steps):
    net.train()
    train_loss = 0
    correct_tr = 0

    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = torch.FloatTensor(inputs).to(device), torch.LongTensor(labels).to(device)
        global_steps += 1

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # print statistics
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        correct_tr += predicted.eq(labels).sum().item()

    net.eval()
    correct_te = 0
    for data, target in testloader:
        data = torch.FloatTensor(data).to(device)
        target = torch.LongTensor(target).to(device)
        output = net(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct_te += pred.eq(target.view_as(pred)).sum().item()

    print('Epoch: {:3d} - loss: {:.3f}, training accuracy: {:.3f}, test accuracy: {:.3f} ----- global_step:{}'.format(
                epoch, train_loss / (i + 1), correct_tr / len(trainloader.dataset),
                correct_te / len(testloader.dataset), global_steps))

    return global_steps, [epoch, train_loss/(i+1), correct_tr/len(trainloader.dataset), correct_te/len(testloader.dataset), global_steps]


epoch = 0
global_steps = 0
tr_stats = []
while True:
    epoch += 1
    global_steps, ep_stats = train(epoch, global_steps)
    tr_stats.append(ep_stats)
    if global_steps >= 64000:
        print('Finished Training')
        break

torch.save(net.state_dict(), 'gl_attn_model.pth.tar')
