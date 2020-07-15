'''
triple networkから次、標準正規分布と経験分布のWasserstein距離の理論計算をLossにしてGeneratorを学習させるぜ
これが本当のマルチタスクラーニング！
'''
import argparse
import os
path = os.getcwd()
path=path[:path.find('timeseries-WGAN')+15]
No = (os.path.basename(__file__))[-4]
# No = str(0) # notebook用
print('実験No.'+No)
import warnings
warnings.simplefilter('ignore')# 警告を非表示
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math
import sys
sys.path.append(path+"/")
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
# 学習用のニューラルネットが置いてあるところ
import models
# p-Wasserstein距離の関数
import Wasserstein

# "output-images"フォルダを作成（既にあるならそれで良し）
os.makedirs("output-images", exist_ok=True)
os.makedirs("parameters", exist_ok=True)

# 真のモデルのパラメータ
phi_ast=[0.3,-0.4,0.2,-0.5,0.6,-0.1,0.1]
p_ast=len(phi_ast)
mu_ast=0
sigma_ast=2

# データセットの作成
trainT=1000
n=100
data_index = range(n)
trainDataSets=[]
for seed in data_index:
    trainData = tsModel.SARIMA(a=phi_ast, N=trainT, random_seed=seed, mu=mu_ast, sigma=sigma_ast)
    trainDataSets.append(trainData)

# 学習する推定モデルの形状や学習方法なんかを決定します
# 学習時のハイパラの決定（入力を受け付ける）
parser = argparse.ArgumentParser()
# ランダムシードについて
parser.add_argument("--generator_seed", type=int, default=0, help="generatorのパラメータの初期値のシード")
parser.add_argument("--discriminator_seed", type=int, default=0, help="discriminatorのパラメータの初期値のシード")
parser.add_argument("--predictor_seed", type=int, default=0, help="predictorのパラメータの初期値のシード")
parser.add_argument("--training_seed", type=int, default=0, help="訓練データを学習させる順番を決めるシード")
parser.add_argument("--data_seed", type=int, default=0, help="Dataの作成に用いる乱数のseed")
# 学習方法について
parser.add_argument("--n_epochs", type=int, default=2000, help="Discriminatorを学習させる回数")
parser.add_argument("--batch_size", type=int, default=64, help="batchの大きさ")
parser.add_argument("--lr", type=float, default=0.00005, help="学習率")
parser.add_argument("--n_critic", type=int, default=5, help="一度generatorを更新するごとに何回discriminatorを更新するか")
parser.add_argument("--discriminator_hidden_unit", type=int, default=64, help="discriminatorの隠れ層のニューロンの数")
# parser.add_argument("--withGP", type=bool, default=True, help="clipingの代わりにGradient Penaltyを加えるかどうか。True/False")
# parser.add_argument("--withCorr", type=bool, default=True, help="Generatorの出力がbatch方向に無相関になるようなロスを加えるかどうか。　True/False")
# モデルの保存やLossの可視化について
parser.add_argument("--sample_interval", type=int, default=100, help="epochを何回まわす度にモデルの保存を行うか")

try:
    opt = parser.parse_args() # .pyの場合はこちらを使用(.ipynbの場合エラーになります)
except:
    opt = parser.parse_args(args=[]) # .ipynbの場合はこちらを使用

print(opt)

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

# 推定モデルの決定
p=7

os.makedirs("output-images/p{0}".format(p), exist_ok=True)
os.makedirs("parameters/p{0}".format(p), exist_ok=True)

torch.manual_seed(opt.generator_seed)
generator = models.LinearGenerator(p = p, input_dim=1, is_bias=False)
torch.manual_seed(opt.predictor_seed)
predictor = models.LinearPredictNet(p=p, input_dim=1, is_bias=True)

# 訓練データを一つ用いて学習させる
dataSeed=opt.data_seed
# こいつをtrain:validation=900:100に分割する
Data = trainDataSets[dataSeed]
Data = torch.tensor(Data, dtype=torch.float)
Data=Data.view(1,-1)
trainData = Data[:,:900]
valData = Data[:,900:]
# trainDataとvalDataを {𝑋𝑡}𝑡0𝑡=𝑡0−𝑝 ごとに取り出しやすいようにMatrixに変換する
trainMatrix = []
for i in range(trainData.shape[1]-(p)):
    ans = trainData[:,i:i+p+1].view(1,Data.shape[0],-1)
    trainMatrix.append(ans)
trainMatrix = torch.cat(trainMatrix)
valMatrix = []
for i in range(valData.shape[1]-(p)):
    ans = valData[:,i:i+p+1].view(1,Data.shape[0],-1)
    valMatrix.append(ans)
valMatrix = torch.cat(valMatrix)

# Optimizers(パラメータに対して定義される)
optimizer_G = torch.optim.RMSprop(params=generator.parameters(), lr=opt.lr)
# optimizer_D = torch.optim.RMSprop(params=discriminator.parameters(), lr=opt.lr)
optimizer_F = torch.optim.RMSprop(params=predictor.parameters(), lr=opt.lr)
# optimizer_F = torch.optim.Adam(params=predictor.parameters())

# 二条誤差MSE
mseLoss = nn.MSELoss()

# パラメータと学習データをGPUに乗っける
generator.to(device)
# discriminator.to(device)
predictor.to(device)
trainMatrix=trainMatrix.to(device)
valMatrix=valMatrix.to(device)
mseLoss.to(device)

saveModel = input('作成したモデルを {0} epochごとに逐次保存しますか？ （Yes：1, No：0）  ----> '.format(opt.sample_interval))
saveModel = bool(int(saveModel))


# 訓練データの時系列のどの時刻を学習に用いるかをランダムにしているが、そのランダムシードを固定する
random.seed(a=opt.training_seed)

do_preTrain = bool(int(input('事前学習をここで行いますか、それとも読み込みますか （行う：1, 読み込む：0）  ----> ')))
pretrain_param = 'parameters/p{0}/No{1}_predictor_epoch{2}_batchSize{3}_DataSeed{4}.pth'.format(p, No, 0, opt.batch_size, dataSeed )
if not do_preTrain:
    try:# モデルパラメータを読み込もうとして失敗したら、それはファイルがないと言うことなので、事前学習をこの場で行う
        predictor.load_state_dict(torch.load(pretrain_param)) 
    except:
        print("モデルが存在しないので事前学習を行います")
        do_preTrain=True

if do_preTrain:
    # ここでまずはFの事前学習を行う
    loss_pre = []
    pretrain_epoch = 1000
    start=time.time()
    for epoch in range(1, pretrain_epoch+1):# epochごとの処理
        # batchの処理は、0~892をランダムに並び替えたリストbatch_sampleを作成し、ここからbatch×(p+1)の学習データを一つづつ獲得する
        l=list(range(trainMatrix.shape[0]-opt.batch_size))
        batch_sample = random.sample(l, len(l))
        for i, batch in enumerate(batch_sample):
            X = trainMatrix[batch : batch+opt.batch_size]# torch.Size([64, 1, 8]) (batch, dim, p+1)
            optimizer_F.zero_grad()
            input_tensor = X[:,:,:-1]
            input_tensor = torch.cat([torch.randn([opt.batch_size,1,1]).to(device), input_tensor], dim=2)
            true_tensor = X[:,:,-1]
            prediction = predictor(input_tensor)
            loss_F = mseLoss(prediction, true_tensor)
            loss_F.backward()
            optimizer_F.step()
        loss_pre.append(loss_F.item())
        print("epoch：{0}/{1}   loss_F：{2: .4f}   経過時間：{3: .1f}秒".format(epoch, pretrain_epoch, round(loss_F.item(), 4), time.time()-start))
        if epoch % 100==0:
            plt.figure(figsize=(13,8))
            plt.title("PredictorのLossの遷移　\n　batchSize:{0}".format(opt.batch_size))
            plt.xlabel("epoch")
            plt.ylabel("Loss")
            plt.plot(loss_pre, label="training")
            # plt.plot(val_loss_F_curve, label="validation")
            plt.legend()
            plt.savefig("preloss.png")
            plt.close()
    torch.save(predictor.state_dict(), pretrain_param)
    print("pre-training終了")

# hat_sigmaに相当する部分がほとんどになってるので1にする
predictor.fc1.weight.data[0][0] = torch.tensor(1)

min_floss=np.inf# epochのflossのの最小値を保管
start=time.time()

batches_done = 0
epoch_done = 0# generatorを学習した回数
loss_curve = []

for epoch in range(1, opt.n_epochs+1):# epochごとの処理(discriminatorのepoch)
    
    # epochごとにbatchで計算したlossを平均した値をloss_curveとして描きたい
    loss_list = []
    
    # batchの処理は、0~892をランダムに並び替えたリストbatch_sampleを作成し、ここからbatch×(p+1)の学習データを一つづつ獲得する
    l=list(range(trainMatrix.shape[0]-opt.batch_size))
    batch_sample = random.sample(l, len(l))
    for i, batch in enumerate(batch_sample):
        
        X = trainMatrix[batch : batch+opt.batch_size]# torch.Size([64, 1, 8]) (batch, dim, p+1)

        # generatorの勾配情報を0に初期化
        optimizer_F.zero_grad()
        optimizer_G.zero_grad()

        # 正規化されたinnoationの推定量をgeneratorを用いて算出
        hat_normeps_t = generator(X)
        # これと過去p時刻の時系列の値（X_{t-1}, .... , X_{t-p}）をpredictorへ入力
        input_tensor = torch.cat([hat_normeps_t.view(opt.batch_size, -1, 1), X[:,:,:-1]], dim=2)
        prediction = predictor(input_tensor)

        loss_G = Variable(Wasserstein.pWasserstein(hat_normeps_t.view(opt.batch_size), p=1), requires_grad=True).to(device)
        loss_F = mseLoss(prediction, X[:,:,-1])

        loss = loss_G+loss_F
        loss_list.append(loss.item())

        # lossを目的関数としてネットワークの全パラメータで微分をしてくれと言う合図
        loss.backward()
        # generatorのパラメータをその微分値とoptimizerを使って更新してくれ！
        optimizer_G.step()
        optimizer_F.step()

        generator_done+=1


    print("epoch：{0}/{1}   batch：{2:003}/{3}   loss_G：{4: .4f}   loss_F：{5: .4f}   経過時間：{6: .1f}秒".format(epoch, opt.n_epochs, i+1, len(batch_sample), round(float(loss_G), 4), round(float(loss_F), 4), time.time()-start))
            
    if saveModel and epoch % opt.sample_interval == 0:
        torch.save(generator.state_dict(), 'parameters/p'+str(p)+'/No{0}_generator_epoch{1}_batchSize{2}_DataSeed{3}.pth'.format(No, epoch, opt.batch_size, dataSeed))
        torch.save(predictor.state_dict(), 'parameters/p'+str(p)+'/No{0}_predictor_epoch{1}_batchSize{2}_DataSeed{3}.pth'.format(No, epoch, opt.batch_size, dataSeed))

    # epochごとにbatchで計算したlossを平均した値をloss_curveとして描きたい
#     try:
#         loss_D_curve.append(sum(loss_D_list)/len(loss_D_list))
#     except:
#         loss_D_curve.append(None)
    try:
        loss_G_curve.append(sum(loss_G_list)/len(loss_G_list))
    except:
        loss_G_curve.append(None)
    try:
        loss_F_curve.append(sum(loss_F_list)/len(loss_F_list))
    except:
        loss_F_curve.append(None)
    

    
    # validationデータによるlossも計算したい
    val_hat_normeps_t = generator(valMatrix)
    val_normeps_t = torch.randn_like(val_hat_normeps_t)
    val_input_tensor = torch.cat([val_hat_normeps_t.view(-1, 1,1), valMatrix[:,:,:-1]], dim=2)
    
#     val_loss_D = -torch.mean(discriminator(val_normeps_t)) + torch.mean(discriminator(val_hat_normeps_t))
#     if opt.withGP:
#         val_loss_D = val_loss_D + gradient_penalty(generated_data=val_hat_normeps_t, real_data=val_normeps_t, gp_weight=gp_weight) 
#     val_loss_D_curve.append(float(val_loss_D))
    val_loss_G = Wasserstein.pWasserstein(val_hat_normeps_t.view(-1), p=1)
    # if opt.withCorr:
    #     val_loss_G = val_loss_G + corr_weight*corr(val_hat_normeps_t)
    val_loss_G_curve.append(float(val_loss_G))
    
    val_loss_F = mseLoss(predictor(val_input_tensor), valMatrix[:,:,0])
    val_loss_F_curve.append(float(val_loss_F))

    # val_loss_Fの最小値を保管
    if min_floss > val_loss_F_curve[-1]:
        min_floss=val_loss_F_curve[-1]
        torch.save(generator.state_dict(), 'parameters/p'+str(p)+'/No{0}_generator_minLoss_batchSize{2}_DataSeed{3}.pth'.format(No, epoch, opt.batch_size, dataSeed))
        torch.save(predictor.state_dict(), 'parameters/p'+str(p)+'/No{0}_predictor_minLoss_batchSize{2}_DataSeed{3}.pth'.format(No, epoch, opt.batch_size, dataSeed))

        print("validationのflossの最小値を更新しました　　Loss:", min_floss)
    
    if epoch % 10==0:
#         plt.figure(figsize=(13,8))
#         plt.title("DiscriminatorのLossの遷移　\n　batchSize:{0}, GPの係数:{1}, Corrの係数:{2}".format(opt.batch_size, gp_weight, corr_weight))
#         plt.xlabel("epoch")
#         plt.ylabel("Loss")
#         plt.plot(loss_D_curve, label="training")
#         plt.plot(val_loss_D_curve, label="validation")
#         plt.legend()
#         plt.savefig("dloss.png")
#         plt.close()

        plt.figure(figsize=(13,8))
        plt.title("GeneratorのLossの遷移　\n　batchSize:{0}".format(opt.batch_size))
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.plot(loss_G_curve, label="training")
        plt.plot(val_loss_G_curve, label="validation")
        plt.legend()
        plt.savefig("gloss.png")
        plt.close()
        
        plt.figure(figsize=(13,8))
        plt.title("PredictorのLossの遷移　\n　batchSize:{0}".format(opt.batch_size))
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.plot(loss_F_curve, label="training")
        plt.plot(val_loss_F_curve, label="validation")
        plt.legend()
        plt.savefig("floss.png")
        plt.close()
    
torch.save(generator.state_dict(), 'parameters/p'+str(p)+'/No{0}_generator_epoch{1}_batchSize{2}_DataSeed{3}.pth'.format(No, epoch, opt.batch_size, dataSeed))
torch.save(predictor.state_dict(), 'parameters/p'+str(p)+'/No{0}_predictor_epoch{1}_batchSize{2}_DataSeed{3}.pth'.format(No, epoch, opt.batch_size, dataSeed))


plt.figure(figsize=(13,8))
plt.title("GeneratorのLossの遷移　\n　batchSize:{0}".format(opt.batch_size))
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.plot(loss_G_curve, label="training")
plt.plot(val_loss_G_curve, label="validation")
plt.legend()
plt.savefig("output-images/p{0}/No{1}_gloss_epoch{2}_batchSize{3}_DataSeed{4}.png".format(p, No, epoch, opt.batch_size, dataSeed ))
# plt.show()
plt.close()

plt.figure(figsize=(13,8))
plt.title("PredictorのLossの遷移　\n　batchSize:{0}".format(opt.batch_size))
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.plot(loss_F_curve, label="training")
plt.plot(val_loss_F_curve, label="validation")
plt.legend()
plt.savefig("output-images/p{0}/No{1}_floss_epoch{2}_batchSize{3}_DataSeed{4}.png".format(p, No, epoch, opt.batch_size, dataSeed ))
# plt.show()
plt.close()