# withoutDiscriminatorを学習させる

import argparse
import os
path = os.getcwd()
path=path[:path.find('estimate-innovation')+20]
No = (os.path.basename(__file__))[-4]
# No = "0" # notebook用
print('実験No.'+No)
import warnings
warnings.simplefilter('ignore')# 警告を非表示
import numpy as np
np.set_printoptions(precision=4, floatmode='fixed', suppress=True)
import matplotlib.pyplot as plt
from scipy import stats
import math
import sys
sys.path.append(path)
import random
import time
import statsmodels.api as sm
from scipy.stats import norm
import japanize_matplotlib
from scipy.stats import gaussian_kde

import torchvision.transforms as transforms
from torchvision.utils import save_image

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
# 人工データを生成してくれる機械が置いてあるところ
import tsModel
# 前処理用の関数の置き場
import my_preprocess
# 学習用のニューラルネットが置いてあるところ
import models
# p-Wasserstein距離の関数
# import Wasserstein

# 学習する推定モデルの形状や学習方法なんかを決定します
# 学習時のハイパラの決定（入力を受け付ける）
parser = argparse.ArgumentParser()
# ランダムシードについて
parser.add_argument("--generator_seed", type=int, default=0, help="generatorのパラメータの初期値のシード")
# parser.add_argument("--discriminator_seed", type=int, default=0, help="discriminatorのパラメータの初期値のシード")
parser.add_argument("--predictor_seed", type=int, default=0, help="predictorのパラメータの初期値のシード")
parser.add_argument("--training_seed", type=int, default=0, help="訓練データを学習させる順番を決めるシード")
parser.add_argument("--data_seed", type=int, default=0, help="Dataの作成に用いる乱数のseed")
# # 学習方法について
parser.add_argument("--n_epochs", type=int, default=2000, help="Generatorを学習させる回数")
parser.add_argument("--batch_size", type=int, default=64, help="batchの大きさ")
parser.add_argument("--sample_interval", type=int, default=100, help="epochを何回まわす度にモデルの保存を行うか")

try:
    opt = parser.parse_args() # .pyの場合はこちらを使用(.ipynbの場合エラーになります)
except:
    opt = parser.parse_args(args=[]) # .ipynbの場合はこちらを使用
print(opt)



# データ生成
phi_a = [1.0, -0.5, 0.7, -0.4]
phi_b = [1.0, -0.7]
p_ast = len(phi_a)
q_ast = len(phi_b)
dataSeed=opt.data_seed
N = 1000
data = tsModel.ARIMA(a=phi_a, b=phi_b, N=N, random_seed=dataSeed, randomness='normal')
inno = tsModel.ARIMA(a=phi_a, b=phi_b, N=N, random_seed=dataSeed, randomness='normal', return_innovation=True)

# 推定に用いる次数
hat_p=5
print("$\hat p$：",hat_p)

os.makedirs("output-images/p{0}".format(hat_p), exist_ok=True)
os.makedirs("parameters/p{0}".format(hat_p), exist_ok=True)

import checkGPU
cuda, device = checkGPU.checkGPU()




def torchJn(n, a, b):
    if n==0:
        ret = torch.sqrt(torch.tensor(np.pi/2))*( torch.erf(b/np.sqrt(2)) - torch.erf(a/np.sqrt(2)) )
        return ret
    elif n==1:
        ret = torch.exp(-a**2/2.0)-torch.exp(-b**2/2.0)
        return ret
    else:
        ret = 0
        if not np.abs(a.item()) == np.inf:
            ret += a**(n-1)*torch.exp(-a**2/2.0)
        if not np.abs(b.item()) == np.inf:
            ret -= b**(n-1)*torch.exp(-b**2/2.0)
        ret += (n-1)*torchJn(n-2, a, b)
        return ret

def combinations_count(n, r):
    return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))
def integral(a, b, c, p):
    ret=0
    if a>=c:
        for i in range(p+1):
            ret += combinations_count(p,i)*(-c)**i/np.sqrt(2*np.pi)*torchJn(n=p-i, a=a, b=b)
    elif a<c and c<b:
        ret = integral(a=a, b=c, c=c, p=p) + integral(a=c, b=b, c=c, p=p)
    else:
        for i in range(p+1):
            ret += combinations_count(p,i)*c**i*(-1)**(p-i)/np.sqrt(2*np.pi)*torchJn(n=p-i, a=a, b=b)
    return ret.to(device)

def pWasserstein(x, p):
    N=x.shape[0]
    ret = 0
    for n in range(1,N+1):
        ret += integral(a=torch.tensor(norm.ppf(q=(n-1)/N, loc=0, scale=1)).to(device), b=torch.tensor(norm.ppf(q=n/N, loc=0, scale=1)).to(device),c=x[n-1],p=p)
    return ret**(1/p)





torch.manual_seed(opt.generator_seed)
generator = models.FullConnectGenerator(p=hat_p, q=0)
torch.manual_seed(opt.predictor_seed)
predictor = models.predictNet(p=hat_p, q=0)

# 訓練データの時系列のどの時刻を学習に用いるかをランダムにしているが、そのランダムシードを固定する
random.seed(a=opt.training_seed)
trainDataSet = my_preprocess.DataSet(tsData=data[:900], p=hat_p)
traindataloader = torch.utils.data.DataLoader(trainDataSet, batch_size=opt.batch_size, shuffle=True, drop_last=True)# dataloaderをiterで回すと、毎回入力と出力のペアがlistでくる
valDataSet = my_preprocess.DataSet(tsData=data[100:], p=hat_p)
valdataloader = torch.utils.data.DataLoader(valDataSet, batch_size=opt.batch_size, shuffle=True, drop_last=True)# dataloaderをiterで回すと、毎回入力と出力のペアがlistでくる


# Optimizers(パラメータに対して定義される)
optimizer_G = torch.optim.RMSprop(params=generator.parameters())
optimizer_F = torch.optim.Adam(params=predictor.parameters(), lr=0.001 )
# 二条誤差MSE
mseLoss = nn.MSELoss()
# GPUに乗っける
generator.to(device)
predictor.to(device)
mseLoss.to(device)

saveModel = input('作成したモデルを {0} epochごとに逐次保存しますか？(本学習の時) （Yes：1, No：0）  ----> '.format(opt.sample_interval))
saveModel = bool(int(saveModel))


do_preTrain = bool(int(input('事前学習をここで行いますか、それとも読み込みますか （行う：1, 読み込む：0）  ----> ')))
pretrain_param = 'parameters/p{0}/No{1}_predictor_epoch{2}_batchSize{3}_DataSeed{4}.pth'.format(hat_p, No, 0, opt.batch_size, dataSeed )
if not do_preTrain:
    try:# モデルパラメータを読み込もうとして失敗したら、それはファイルがないと言うことなので、事前学習をこの場で行う
        predictor.load_state_dict(torch.load(pretrain_param)) 
    except:
        print("モデルが存在しないので事前学習を行います")
        do_preTrain=True

if do_preTrain:
    # ここでまずはFの事前学習を行う
    loss_pre = []
    val_loss_pre = []
    pretrain_epoch = 1000
    start=time.time()
    for epoch in range(1, pretrain_epoch+1):# epochごとの処理
        epoch_loss = 0
        for X, Y in traindataloader:
            input_tensor = torch.cat([torch.randn([opt.batch_size,1]).to(device), X.to(device)], axis=1)# ランダムな次元を追加
            optimizer_F.zero_grad()
            loss_F = mseLoss(predictor(input_tensor), Y.to(device))
            loss_F.backward()
            optimizer_F.step()
            epoch_loss += loss_F.item()
        loss_pre.append(epoch_loss/len(traindataloader))
        
        epoch_loss = 0
        for X, Y in valdataloader:
            input_tensor = torch.cat([torch.ones([opt.batch_size,1]).to(device), X.to(device)], axis=1)# ランダムな次元を追加
            loss_F = mseLoss(predictor(input_tensor), Y.to(device))
            epoch_loss += loss_F.item()
        val_loss_pre.append(epoch_loss/len(valdataloader))
        
        print("epoch：{0}/{1}   loss_F：{2: .4f}   経過時間：{3: .1f}秒".format(epoch, pretrain_epoch, round(loss_F.item(), 4), time.time()-start))
        if epoch % 100==0:
            plt.figure(figsize=(13,8))
            plt.title("PredictorのLossの遷移　\n　batchSize:{0}".format(opt.batch_size))
            plt.xlabel("epoch")
            plt.ylabel("Loss")
            plt.plot(loss_pre, label="training")
            plt.plot(val_loss_pre, label="validation")
            plt.legend()
            plt.savefig("preloss.png")
            plt.close()
    torch.save(predictor.state_dict(), pretrain_param)
    
    plt.figure(figsize=(13,8))
    plt.title("PredictorのLossの遷移　\n　batchSize:{0}".format(opt.batch_size))
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.plot(loss_pre, label="training")
    plt.plot(val_loss_pre, label="validation")
    plt.legend()
    plt.savefig("output-images/p{0}/No{1}_preloss_epoch{2}_batchSize{3}_DataSeed{4}.png".format(hat_p, No, epoch, opt.batch_size, dataSeed ))
    plt.close()
    print("pre-training終了")
print("モデルの保存先：",pretrain_param)


trainDataSet = my_preprocess.DataSet_forGandF(tsData=data[:900], p=hat_p)
traindataloader = torch.utils.data.DataLoader(trainDataSet, batch_size=opt.batch_size, shuffle=True, drop_last=True)# dataloaderをiterで回すと、毎回入力と出力のペアがlistでくる
valDataSet = my_preprocess.DataSet_forGandF(tsData=data[900:], p=hat_p)
valdataloader = torch.utils.data.DataLoader(valDataSet, batch_size=opt.batch_size, shuffle=True, drop_last=True)# dataloaderをiterで回すと、毎回入力と出力のペアがlistでくる

min_floss=np.inf# epochのflossのの最小値を保管
start=time.time()
# epochごとにbatchで計算したlossを平均した値をloss_curveとして描きたい
loss_Wasserstein = []
loss_mse = []
val_loss_Wasserstein = []
val_loss_mse = []

for epoch in range(1, opt.n_epochs+1):# epochごとの処理(discriminatorのepoch)

    
    epoch_loss_Wasserstein = 0
    epoch_loss_mse = 0
    for X, Y in traindataloader:
        X = X.to(device)
        Y = Y.to(device)
        # Wasserstein距離に基づくGeneratorの学習
        loss_G = pWasserstein(generator(X), p=1)
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()
        epoch_loss_Wasserstein += loss_G.item()
        # Fの学習
        input_tensor = torch.cat([generator(X), X[:,:-1]], axis=1)
        loss_F = mseLoss(predictor(input_tensor), Y)
        optimizer_G.zero_grad()
        optimizer_F.zero_grad()
        loss_F.backward()
        optimizer_F.step()
        optimizer_G.step()
        epoch_loss_mse += loss_F.item()
    loss_Wasserstein.append(epoch_loss_Wasserstein/len(traindataloader))
    loss_mse.append(epoch_loss_mse/len(traindataloader))
    
    epoch_loss_Wasserstein = 0
    epoch_loss_mse = 0
    for X, Y in traindataloader:
        X = X.to(device)
        Y = Y.to(device)
        loss_G = pWasserstein(generator(X), p=1)
        epoch_loss_Wasserstein += loss_G.item()
        input_tensor = torch.cat([generator(X), X[:,:-1]], axis=1)
        loss_F = mseLoss(predictor(input_tensor), Y.to(device))
        epoch_loss_mse += loss_F.item()
    val_loss_Wasserstein.append(epoch_loss_Wasserstein/len(valdataloader))
    val_loss_mse.append(epoch_loss_mse/len(valdataloader))
    
    print("epoch：{0}/{1}   loss_Wass：{2: .4f}    loss_mse：{3: .4f}     経過時間：{4: .1f}秒".format(epoch, opt.n_epochs, round(loss_Wasserstein[-1], 4), round(loss_mse[-1], 4), time.time()-start))
             
    if saveModel and epoch % opt.sample_interval == 0:
        torch.save(generator.state_dict(), 'parameters/p'+str(hat_p)+'/No{0}_generator_epoch{1}_batchSize{2}_DataSeed{3}.pth'.format(No, epoch, opt.batch_size, dataSeed))
        torch.save(predictor.state_dict(), 'parameters/p'+str(hat_p)+'/No{0}_predictor_epoch{1}_batchSize{2}_DataSeed{3}.pth'.format(No, epoch, opt.batch_size, dataSeed))
  
    if epoch % 100==0:

        plt.figure(figsize=(13,8))
        plt.title("Wasserstein距離のLossの遷移　\n　batchSize:{0}".format(opt.batch_size))
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.plot(loss_Wasserstein, label="training")
        plt.plot(val_loss_Wasserstein, label="validation")
        plt.legend()
        plt.savefig("gloss.png")
        plt.close()
        
        plt.figure(figsize=(13,8))
        plt.title("PredictorのLossの遷移　\n　batchSize:{0}".format(opt.batch_size))
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.plot(loss_mse, label="training")
        plt.plot(val_loss_mse, label="validation")
        plt.legend()
        plt.savefig("floss.png")
        plt.close()
#     break
torch.save(generator.state_dict(), 'parameters/p'+str(hat_p)+'/No{0}_generator_epoch{1}_batchSize{2}_DataSeed{3}.pth'.format(No, epoch, opt.batch_size, dataSeed))
torch.save(predictor.state_dict(), 'parameters/p'+str(hat_p)+'/No{0}_predictor_epoch{1}_batchSize{2}_DataSeed{3}.pth'.format(No, epoch, opt.batch_size, dataSeed))

plt.figure(figsize=(13,8))
plt.title("Wasserstein距離のLossの遷移　\n　batchSize:{0}".format(opt.batch_size))
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.plot(loss_Wasserstein, label="training")
plt.plot(val_loss_Wasserstein, label="validation")
plt.legend()
plt.savefig("output-images/p{0}/No{1}_gloss_epoch{2}_batchSize{3}_DataSeed{4}.png".format(hat_p, No, epoch, opt.batch_size, dataSeed ))
plt.close()

plt.figure(figsize=(13,8))
plt.title("PredictorのLossの遷移　\n　batchSize:{0}".format(opt.batch_size))
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.plot(loss_mse, label="training")
plt.plot(val_loss_mse, label="validation")
plt.legend()
plt.savefig("output-images/p{0}/No{1}_floss_epoch{2}_batchSize{3}_DataSeed{4}.png".format(hat_p, No, epoch, opt.batch_size, dataSeed ))
plt.close()
