import torch
import numpy as np
from global_attention_backup import GlobalAttentionNetwork
import torchvision
import torchvision.transforms as transforms
import os

path = './'
device = 'cuda:0'
c_path = '/data/cifar10_c/'
p_path = '/data/cifar10_p/'

c_files = os.listdir(c_path)
c_files.remove('labels.npy')
p_files = os.listdir(p_path)

labels = np.load(c_path+'labels.npy')
labels = torch.LongTensor(labels).to(device)


class CorrLoader():
    def __init__(self, file, batch):
        data = np.load(file)
        data = data / 256
        data = (torch.FloatTensor(data).to(device) - torch.FloatTensor([0.4914, 0.4822, 0.4465]).to(device)) / torch.FloatTensor(
            [0.2023, 0.1994, 0.2010]).to(device)
        self.data = data.permute(0, 3, 1, 2)
        self.batch = batch
        self.iter = 0
        self.end = False

    def get_next_batch(self):
        if not self.end:
            batch = self.data[self.iter:self.iter+self.batch]
            labs = labels[self.iter:self.iter+self.batch]
            self.iter += self.batch
            if self.iter >= len(self.data):
                self.end = True
            return batch, labs
        else:
            return

    def reset(self):
        self.iter = 0
        return


class PertLoader():
    def __init__(self, file):
        data = np.load(file)
        data = data / 256
        data = (torch.FloatTensor(data) - torch.FloatTensor([0.4914, 0.4822, 0.4465])) / torch.FloatTensor(
            [0.2023, 0.1994, 0.2010])
        self.data = data.permute(0, 1, 4, 2, 3)
        self.iter = 0
        self.end = False

    def get_next_batch(self):
        if not self.end:
            batch = self.data[self.iter].squeeze()
            labs = labels[self.iter].expand(batch.shape[0])
            self.iter += 1
            if self.iter >= len(self.data):
                self.end = True
            return batch, labs
        else:
            return

    def reset(self):
        self.iter = 0
        return


net = GlobalAttentionNetwork()
net.load_state_dict(torch.load('gl_attn_model_b.pth.tar'))
net.to(device)
net.eval()

total_count = 0
# for file in c_files:
#     loader = CorrLoader(c_path+file, 50)
#     while not loader.end:
#         test, labs = loader.get_next_batch()
#         output = net(test)
#         pred = output.argmax(dim=1)
#         count = pred.eq(labs).sum().item()
#         total_count += count
#         print(file, loader.iter, count)
#
c_acc = total_count / (50000 * 19)

total_count = 0
for file in p_files:
    loader = PertLoader(p_path+file)
    while not loader.end:
        test, labs = loader.get_next_batch()
        test = test.to(device)
        output = net(test)
        pred = output.argmax(dim=1)
        count = pred.eq(labs).sum().item()
        total_count += count
        print(file, loader.iter, count)
p_acc = total_count / (50000 * 31 * 20)

print(c_acc, p_acc)



