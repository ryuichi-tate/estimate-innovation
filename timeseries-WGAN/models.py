import torch.nn as nn
import torch.nn.functional as F
import torch

class Generator(nn.Module):
    def __init__(self, n_filter=8, generator_kernel_size=5, p=7, q=3): # ネットワークで使う関数を定義する。
        super(Generator, self).__init__()
        # kernel
        # nn.Conv1dについては、  https://pytorch.org/docs/stable/nn.html#conv1d  を参照
        self.conv1 = nn.Conv1d(1, n_filter, kernel_size =generator_kernel_size, stride =1, padding=(generator_kernel_size-1)//2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv2 = nn.Conv1d(n_filter, n_filter, kernel_size =generator_kernel_size, stride =1, padding=(generator_kernel_size-1)//2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        # self.conv3 = nn.Conv1d(n_filter, n_filter, kernel_size =generator_kernel_size, stride =1, padding=(generator_kernel_size-1)//2, dilation=1, groups=1, bias=True, padding_mode='zeros')
        # self.batchnorm1 = nn.BatchNorm1d(n_filter)
        # self.batchnorm2 = nn.BatchNorm1d(n_filter)
        # self.batchnorm3 = nn.BatchNorm1d(n_filter)
        # 線形変換: y = Wx + b
        self.fc1 = nn.Linear(n_filter*(p+1), q+1)
            
    def forward(self, x):# ここでネットワークを構成する。入力はx。
        x = self.conv1(x)
        # x = self.batchnorm1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        # x = self.batchnorm2(x)
        x = F.leaky_relu(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        return x # 出力

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class Discriminator(nn.Module):
    def __init__(self,q=3,discriminator_hidden_unit=64):
        super(Discriminator, self).__init__()
        
        self.sigma = nn.Parameter(torch.tensor([1.0]))
        self.fc1 = nn.Linear(q+1, discriminator_hidden_unit)
        self.fc2 = nn.Linear(discriminator_hidden_unit, discriminator_hidden_unit)        
        self.fc3 = nn.Linear(discriminator_hidden_unit, discriminator_hidden_unit)        
        self.fc4 = nn.Linear(discriminator_hidden_unit, 1)        
        
    def forward(self, x, is_from_generator=True):
        if not is_from_generator:# 標準正規分布からのサンプリングが入力の場合は定数倍する
            x = self.sigma*x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x

class LinearGenerator(nn.Module):
    def __init__(self, p=7, input_dim=1, is_bias=False): # ネットワークで使う関数を定義する。
        super(LinearGenerator, self).__init__()
        # 線形変換: y = Wx + b
        self.fc1 = nn.Linear((p+1)*input_dim, 1,bias=is_bias)
            
    def forward(self, x):# ここでネットワークを構成する。入力はx。
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        return x # 出力

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class predictNet(nn.Module):
    def __init__(self, p=7, q=3, n_unit1=16, n_unit2=32):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(p+q+1, n_unit1)
        self.fc2 = nn.Linear(n_unit1, n_unit2)
        self.fc3 = nn.Linear(n_unit2,1)
        
    def forward(self,x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        return x

class LinearPredictNet(nn.Module):
    def __init__(self, p=7, input_dim=1, is_bias=False): # ネットワークで使う関数を定義する。
        super(LinearPredictNet, self).__init__()
        # 線形変換: y = Wx + b
        self.fc1 = nn.Linear((p+1)*input_dim, 1,bias=is_bias)
            
    def forward(self, x):# ここでネットワークを構成する。入力はx。
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        return x # 出力

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features