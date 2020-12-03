import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet_model import ResidualBlock


class LayerNorm(nn.Module):
    def __init__(self, channels):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        out = x.transpose(1, -1)
        out = self.norm(out)
        out = out.transpose(1, -1)
        return out


class GlobalAttention(nn.Module):
    def __init__(self, in_c, out_c, n_head, dims, bias=False):
        super(GlobalAttention, self).__init__()

        assert out_c % n_head == 0, "out_c must be divisible by n_head"
        assert (out_c // n_head) % 2 == 0, "channel dimension for each head must be even"
        self.out_c = out_c
        self.n_head = n_head
        self.head_c = out_c // n_head
        self.rel_c = self.head_c // 2

        self.rel_dim = dims * 2 - 1
        self.rel_x = nn.Parameter(torch.randn(self.n_head, self.rel_c, self.rel_dim, 1), requires_grad=True)
        self.rel_y = nn.Parameter(torch.randn(self.n_head, self.rel_c, 1, self.rel_dim), requires_grad=True)

        self.q_conv = nn.Conv2d(in_c, out_c, kernel_size=1, bias=bias)
        self.k_conv = nn.Conv2d(in_c, out_c, kernel_size=1, bias=bias)
        self.v_conv = nn.Conv2d(in_c, out_c, kernel_size=1, bias=bias)

    def forward(self, x):
        batch, in_c, height, width = x.size()
        q = self.q_conv(x)
        v = self.v_conv(x)
        k = self.k_conv(x)

        q = q.view(batch, self.n_head, self.head_c, height, width, 1, 1)
        k = k.view(batch, self.n_head, self.head_c, 1, 1, height, width)
        v = v.view(batch, self.n_head, self.head_c, 1, 1, height, width)

        pos_rel = torch.cat((self.rel_x.expand(self.n_head, self.rel_c, self.rel_dim, self.rel_dim),
                            self.rel_y.expand(self.n_head, self.rel_c, self.rel_dim, self.rel_dim)), dim=1)
        pos_rel = pos_rel.unfold(2, height, 1).unfold(3, width, 1).flip(4, 5)
        k = k.expand(batch, self.n_head, self.head_c, height, width, height, width).contiguous()
        k += pos_rel

        qk = (q * k).sum([2])
        soft_qk = F.softmax(qk.flatten(4, 5), -1)
        soft_qk = soft_qk.view(batch, self.n_head, 1, height, width, height, width)

        out = (soft_qk * v).sum([5, 6])
        out = out.view(batch, self.out_c, height, width)
        return out
    
    
class GlobalAttentionWithDecay(nn.Module):
    def __init__(self, in_c, out_c, n_head, dims, k, bias=False):
        super(GlobalAttentionWithDecay, self).__init__()

        assert out_c % n_head == 0, "out_c must be divisible by n_head"
        assert (out_c // n_head) % 2 == 0, "channel dimension for each head must be even"
        self.out_c = out_c
        self.n_head = n_head
        self.head_c = out_c // n_head
        self.rel_c = self.head_c // 2

        self.rel_dim = dims * 2 - 1
        self.rel_x = nn.Parameter(torch.randn(self.n_head, self.rel_c, self.rel_dim, 1), requires_grad=True)
        self.rel_y = nn.Parameter(torch.randn(self.n_head, self.rel_c, 1, self.rel_dim), requires_grad=True)

        i, j = np.indices((dims * 2 - 1, dims * 2 - 1)) - dims + 1
        dec = np.maximum(np.abs(i), np.abs(j))
        dec = torch.FloatTensor(dec)

        self.register_buffer('dec', dec)

        self.k = nn.Parameter(torch.FloatTensor([k]), requires_grad=True)
        self.q_conv = nn.Conv2d(in_c, out_c, kernel_size=1, bias=bias)
        self.k_conv = nn.Conv2d(in_c, out_c, kernel_size=1, bias=bias)
        self.v_conv = nn.Conv2d(in_c, out_c, kernel_size=1, bias=bias)

    def forward(self, x):
        batch, in_c, height, width = x.size()
        q = self.q_conv(x)
        v = self.v_conv(x)
        k = self.k_conv(x)

        q = q.view(batch, self.n_head, self.head_c, height, width, 1, 1)
        k = k.view(batch, self.n_head, self.head_c, 1, 1, height, width)
        v = v.view(batch, self.n_head, self.head_c, 1, 1, height, width)


        pos_rel = torch.cat((self.rel_x.expand(self.n_head, self.rel_c, self.rel_dim, self.rel_dim),
                            self.rel_y.expand(self.n_head, self.rel_c, self.rel_dim, self.rel_dim)), dim=1)
        pos_rel = pos_rel.unfold(2, height, 1).unfold(3, width, 1).flip(4, 5)
        k = k.expand(batch, self.n_head, self.head_c, height, width, height, width).contiguous()
        k += pos_rel

        qk = (q * k).sum([2])
        dec = 1 / ((1. + torch.abs(self.k)) ** self.dec)
        qk *= dec.unfold(0, height, 1).unfold(1, width, 1).flip(2, 3)
        soft_qk = F.softmax(qk.flatten(4, 5), -1)
        soft_qk = soft_qk.view(batch, self.n_head, 1, height, width, height, width)

        out = (soft_qk * v).sum([5, 6])
        out = out.view(batch, self.out_c, height, width)
        return out
    
    
class GlobalAttentionWithTanh(nn.Module):
    def __init__(self, in_c, out_c, n_head, dims, k, bias=False):
        super(GlobalAttentionWithTanh, self).__init__()

        assert out_c % n_head == 0, "out_c must be divisible by n_head"
        assert (out_c // n_head) % 2 == 0, "channel dimension for each head must be even"
        self.out_c = out_c
        self.n_head = n_head
        self.head_c = out_c // n_head
        self.rel_c = self.head_c // 2

        self.rel_dim = dims * 2 - 1
        self.rel_x = nn.Parameter(torch.randn(self.n_head, self.rel_c, self.rel_dim, 1), requires_grad=True)
        self.rel_y = nn.Parameter(torch.randn(self.n_head, self.rel_c, 1, self.rel_dim), requires_grad=True)

        i, j = np.indices((dims * 2 - 1, dims * 2 - 1)) - dims + 1
        dec = np.maximum(np.abs(i), np.abs(j))
        dec = torch.FloatTensor(dec)

        self.register_buffer('dec', dec)

        self.k = nn.Parameter(torch.FloatTensor([k]), requires_grad=False)
        self.q_conv = nn.Conv2d(in_c, out_c, kernel_size=1, bias=bias)
        self.k_conv = nn.Conv2d(in_c, out_c, kernel_size=1, bias=bias)
        self.v_conv = nn.Conv2d(in_c, out_c, kernel_size=1, bias=bias)

    def forward(self, x):
        batch, in_c, height, width = x.size()
        q = self.q_conv(x)
        v = self.v_conv(x)
        k = self.k_conv(x)

        q = q.view(batch, self.n_head, self.head_c, height, width, 1, 1)
        k = k.view(batch, self.n_head, self.head_c, 1, 1, height, width)
        v = v.view(batch, self.n_head, self.head_c, 1, 1, height, width)


        pos_rel = torch.cat((self.rel_x.expand(self.n_head, self.rel_c, self.rel_dim, self.rel_dim),
                            self.rel_y.expand(self.n_head, self.rel_c, self.rel_dim, self.rel_dim)), dim=1)
        pos_rel = pos_rel.unfold(2, height, 1).unfold(3, width, 1).flip(4, 5)
        k = k.expand(batch, self.n_head, self.head_c, height, width, height, width).contiguous()
        k += pos_rel

        qk = (q * k).sum([2])
        dec = torch.tanh(-5*(self.dec-(10*self.k)))/2 + 0.5
        qk *= dec.unfold(0, height, 1).unfold(1, width, 1).flip(2, 3)
        soft_qk = F.softmax(qk.flatten(4, 5), -1)
        soft_qk = soft_qk.view(batch, self.n_head, 1, height, width, height, width)

        out = (soft_qk * v).sum([5, 6])
        out = out.view(batch, self.out_c, height, width)
        return out    

class GlobalAttentionLayer(nn.Module):
    def __init__(self, in_c, out_c, n_head, dims, ff=None, norm='bn', bias=False):
        super(GlobalAttentionLayer, self).__init__()

        self.attn = GlobalAttention(in_c, out_c, n_head, dims, bias=bias)

        if norm == 'bn':
            self.norm1 = nn.BatchNorm2d(out_c)
            self.norm2 = nn.BatchNorm2d(out_c)
        elif norm == 'ln':
            self.norm1 = LayerNorm(out_c)
            self.norm2 = LayerNorm(out_c)
        else:
            raise AssertionError("Use 'bn' for batch norm and 'ln' for layer norm")

        if ff is None:
            ff = out_c * 2

        self.conv1 = nn.Conv2d(out_c, ff, kernel_size=1)
        self.bn_ff = nn.BatchNorm2d(ff)
        self.conv2 = nn.Conv2d(ff, out_c, kernel_size=1)

    def forward(self, x):
        out = self.attn(x)
        res = x
        if x.shape != out.shape:
            assert x.shape[1] < out.shape[1], "out_c must be larger than in_c"
            res = F.pad(res, [0, 0, 0, 0, 0, out.shape[1] - x.shape[1]])
        out += res
        out = self.norm1(out)
        out = F.relu(out)

        res = out
        out = self.conv1(out)
        out = self.bn_ff(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += res
        out = self.norm2(out)
        return out


class GlobalAttentionNetwork(nn.Module):
    def __init__(self):
        super(GlobalAttentionNetwork, self).__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=[1, 1])
        # self.conv = nn.Conv2d(3, 8, kernel_size=1)
        self.bn = nn.BatchNorm2d(8)

        self.res = ResidualBlock(8, 32, stride=2, down_sample=True)

        self.attn1_1 = GlobalAttentionLayer(32, 64, 8, 16)
        self.attn1_2 = GlobalAttentionLayer(64, 64, 8, 16)
        self.pool1 = nn.AvgPool2d(2)

        self.attn2_1 = GlobalAttentionLayer(64, 128, 8, 8)
        self.attn2_2 = GlobalAttentionLayer(128, 128, 8, 8)
        self.pool2 = nn.AvgPool2d(8)

        #self.attn3_1 = GlobalAttentionLayer(64, 128, 8, 4)
        #self.attn3_2 = GlobalAttentionLayer(128, 128, 8, 4)
        #self.pool3 = nn.AvgPool2d(4)

        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.res(out)
        out = self.attn1_1(out)
        out = self.attn1_2(out)
        out = self.pool1(out)
        out = self.attn2_1(out)
        out = self.attn2_2(out)
        out = self.pool2(out)
        #out = self.attn3_1(out)
        #out = self.attn3_2(out)
        #out = self.pool3(out)
        out = out.flatten(1, 3)
        out = self.fc(out)
        return out
