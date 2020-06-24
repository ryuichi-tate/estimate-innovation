import argparse
import os
path = os.getcwd()
path=path[:path.find('timeseries-WGAN')+15]
import numpy as np
from scipy import stats
import math
import sys
sys.path.append(path+"/")
import random
import statsmodels.api as sm
from scipy.stats import norm

import torchvision.transforms as transforms
from torchvision.utils import save_image

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

# 人工データを生成してくれる機械が置いてあるところ
import tsModel
# 学習用のニューラルネットが置いてあるところ
import models

import matplotlib.pyplot as plt
import japanize_matplotlib


# フォルダを作成（既にあるならそれで良し）
os.makedirs("output-images", exist_ok=True)
os.makedirs("images", exist_ok=True)
os.makedirs("parameters", exist_ok=True)



# 学習時のハイパラの決定（入力を受け付ける）
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20000, help="学習させる回数")
parser.add_argument("--network_seed", type=int, default=0, help="networkのパラメータの初期値のシード")
parser.add_argument("--random_seed", type=int, default=0, help="訓練データの時系列のどの時刻を学習に用いるかをランダムに決定する時のシード")
parser.add_argument("--batch_size", type=int, default=64, help="batchの大きさ")
parser.add_argument("--lr", type=float, default=0.00005, help="学習率")
parser.add_argument("--sample_interval", type=int, default=1000, help="batchを何回学習させる度にgeneratorの出力を保存するか")
parser.add_argument("--network_bias", type=bool, default=False, help="networkにbiasを入れるかどうか。True/False")
parser.add_argument("--data_seed", type=int, default=0, help="Dataの作成に用いる乱数のseed")
opt = parser.parse_args()

p=7

phi=[0.3, -0.4, 0.2, -0.5, 0.6, -0.1, 0.1]

Data = tsModel.SARIMA(a=phi, N=1400, random_seed=opt.data_seed, sigma=2)
Data = torch.tensor(Data, dtype=torch.float)
Data = torch.tensor(Data)
plt.figure(figsize=(13,8))
plt.plot(Data)
plt.title("ARモデルの人工データその"+str(opt.data_seed)+"\n$\phi^{\\ast}="+str(phi)+", \sigma^{\\ast}=2$")
plt.savefig("images/AR7モデルの人工データその{0}.png".format(opt.data_seed))
plt.close()


Data=Data.view(1,-1)
trainData = Data[:,:1000]
valData = Data[:,1000:1200]
testData = Data[:,1200:]


trainMatrix = []
for i in range(trainData.shape[1]-(p+1)):
    ans = trainData[:,i:i+p+1].view(1,Data.shape[0],-1)
    trainMatrix.append(ans)
trainMatrix = torch.cat(trainMatrix)



valMatrix = []
for i in range(valData.shape[1]-(p+1)):
    ans = valData[:,i:i+p+1].view(1,Data.shape[0],-1)
    valMatrix.append(ans)
valMatrix = torch.cat(valMatrix)


torch.manual_seed(opt.network_seed)
net = models.LinearPredictNet(p = p, input_dim=1, is_bias=opt.network_bias)


# gpuが使えるかどうか
cuda = True if torch.cuda.is_available() else False
if cuda:
    print("GPUが使えます。")
    use_gpu = input('GPUを使いますか？ （Yes：1, No：0）  ----> ')
    cuda = bool(int(use_gpu))
else:
    print("GPUは使えません。")
    
if cuda:
    gpu_id = input('使用するGPUの番号を入れてください : ')
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
device = torch.device('cuda:'+gpu_id if cuda else 'cpu')


# Optimizers(パラメータに対して定義される)
optimizer = torch.optim.RMSprop(net.parameters(), lr=opt.lr)


# パラメータと学習データをGPUに乗っける
net.to(device)

trainMatrix=trainMatrix.to(device)
valMatrix=valMatrix.to(device)



saveModel = input('作成したモデルを保存しますか？ （Yes：1, No：0）  ----> ')
saveModel = bool(int(saveModel))


import japanize_matplotlib
from scipy.stats import gaussian_kde


train_loss_curve = []
val_loss_curve = []

for epoch in range(1, opt.n_epochs+1):# epochごとの処理
    for i, batch in enumerate(range(0, trainMatrix.shape[0]-opt.batch_size, opt.batch_size)):# batchごとの処理
        
        # generatorへの入力を用意する
        X = trainMatrix[batch:batch+opt.batch_size]# torch.Size([64, 1, 8])
        # 時系列の順番はそのまま入力した方がいいのかな？
        rand=random.randint(0,trainMatrix.shape[0] - trainMatrix.shape[0]// opt.batch_size*opt.batch_size)
        X = trainMatrix[batch+rand : batch+rand+opt.batch_size]# torch.Size([64, 1, 8])
    
        
        input_tensor = X[:,:,0:p]
        true_tensor = X[:,:,p:p+1].view(opt.batch_size, -1)
        input_tensor = torch.cat([input_tensor, torch.randn([opt.batch_size,1,1]).to(device)], dim=2)
        input_tensor = Variable(input_tensor)
        
        Loss = nn.MSELoss()
        output_tensor = net(input_tensor)
        loss = Loss(output_tensor, true_tensor)

        loss.backward()
        optimizer.step()
    
    train_loss_curve.append(loss.item())
    
    val_input = torch.cat([valMatrix[:,:,0:p], torch.randn([valMatrix.shape[0],1,1]).to(device)], dim=2)
    val_target = valMatrix[:,:,p:p+1].view(valMatrix.shape[0], -1)
    val_loss = Loss(net(val_input), val_target)
    
    val_loss_curve.append(val_loss.item())
    
    if saveModel:
        if epoch%opt.sample_interval==0:
            torch.save(net.state_dict(), 'parameters/network_epoch{0}_{1}_batchSize{2}_networkSeed{3}_p{4}_networkBias{5}_dataSeed{6}.pth'.format(epoch, opt.n_epochs, opt.batch_size, opt.network_seed, p, opt.network_bias, opt.data_seed))



    if epoch%opt.sample_interval==0 or epoch==opt.n_epochs:
        plt.figure(figsize=(13,8))
        plt.title("Lossの遷移　\n　epoch:{0}, batchSize:{2}, network initial Seed:{3}, p:{4}, bias:{5}, data Seed:{6}".format(epoch, opt.n_epochs, opt.batch_size, opt.network_seed, p, opt.network_bias, opt.data_seed))
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.plot(train_loss_curve, label="training")
        plt.plot(val_loss_curve, label="validation")
        plt.legend()
        plt.savefig("loss.png")
        plt.close()
    
    print("[Epoch %d/%d] [train loss: %f] [validation loss: %f]" % (epoch, opt.n_epochs,loss.item(), val_loss.item()))


torch.save(net.state_dict(), 'parameters/network_epoch{1}_batchSize{2}_networkSeed{3}_p{4}_networkBias{5}_dataSeed{6}.pth'.format(epoch, opt.n_epochs, opt.batch_size, opt.network_seed, p, opt.network_bias, opt.data_seed))


plt.figure(figsize=(13,8))
plt.title("Lossの遷移　\n　epoch:{1}, batchSize:{2}, network initial Seed:{3}, p:{4}, bias:{5}, data Seed:{6}".format(epoch, opt.n_epochs, opt.batch_size, opt.network_seed, p, opt.network_bias, opt.data_seed))
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.plot(train_loss_curve, label="training")
plt.plot(val_loss_curve, label="validation")
plt.legend()
plt.savefig("output-images/loss_epoch{1}_batchSize{2}_networkSeed{3}_p{4}_networkBias{5}_dataSeed{6}.png".format(epoch, opt.n_epochs, opt.batch_size, opt.network_seed, p, opt.network_bias, opt.data_seed))
plt.close()