
'''
過去p時刻の実現値と過去q時刻のiidなノイズから次時刻の値が決定される時系列モデルをニューラルネットで学習する
innovation系列を推定する部分Gと、次時刻を予測する部分Fと、Gの出力を独立な正規分布に寄せる部分Dからなる
WassersteinGANを用いてGの出力を正規分布に近づける。
ニューラルネットを用いて作成された人工データを用いて結果を見てみる
具体的には、Shapiro-wilk検定のp値と相互相関係数の、学習に伴う遷移を観察する
'''


# まずはimport
import argparse
import os
path = os.getcwd()
path=path[:path.find('timeseries-WGAN')+15]
import numpy as np
from scipy import stats
from scipy.stats import norm
import math
import sys
sys.path.append(path+"/")
import random

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



# 結果を出力するためのフォルダを作成（既にあるならそれで良し）
os.makedirs("output-images", exist_ok=True)
os.makedirs("parameters", exist_ok=True)



# 学習時のハイパラの決定（入力を受け付ける）
# 書き方は、$ python hoge.py --n_epoch 1000 --batch_size 10 という感じ
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20000, help="Discriminatorを学習させる回数")
parser.add_argument("--batch_size", type=int, default=64, help="batchの大きさ")
parser.add_argument("--lr", type=float, default=0.00005, help="学習率")
parser.add_argument("--random_seed", type=int, default=0, help="訓練データの時系列のどの時刻を学習に用いるかをランダムに決定する時のシード")
parser.add_argument("--n_cpu", type=int, default=os.cpu_count(), help="number of cpu threads to use during batch generation")
parser.add_argument("--p", type=int, default=7, help="ARの次数(generatorへの入力の次元)")
parser.add_argument("--q", type=int, default=3, help="MAの次数(generatorからの出力の次元)")
parser.add_argument("--generator_seed", type=int, default=0, help="generatorのパラメータの初期値のシード")
parser.add_argument("--discriminator_seed", type=int, default=0, help="discriminatorのパラメータの初期値のシード")
parser.add_argument("--generator_kernel_size", type=int, default=5, help="generatorの中の畳み込みフィルタの大きさ（必ず奇数にすること）")
parser.add_argument("--n_filter", type=int, default=8, help="generatorの最初に生データにかけるフィルタの数")
parser.add_argument("--discriminator_hidden_unit", type=int, default=64, help="discriminatorの隠れ層のニューロンの数")
parser.add_argument("--n_critic", type=int, default=5, help="一度generatorをbackpropするごとに何回discriminatorをbackpropするか")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="batchを何回学習させる度にgeneratorの出力を保存するか")
parser.add_argument("--withGP", type=bool, default=False, help="clipingの代わりにGradient Penaltyを加えるかどうか。True/False")
opt = parser.parse_args()
print(opt)



# gpuが使えるかどうか
cuda = True if torch.cuda.is_available() else False
print("GPUは使えますか？ (True/False)  ----> ",cuda)

if cuda:
    gpu_id = input('使用するGPUの番号を入れてください : ')
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
device = torch.device('cuda:'+gpu_id if torch.cuda.is_available() else 'cpu')



# ネットワークのインスタンスを作成
# generator用のネットワークを作成
torch.manual_seed(opt.generator_seed)
generator = models.Generator(n_filter=opt.n_filter, generator_kernel_size=opt.generator_kernel_size, p=opt.p, q=opt.q)
print(generator)

# discriminator用のネットワークを作成
torch.manual_seed(opt.discriminator_seed)
discriminator = models.Discriminator(q=opt.q, discriminator_hidden_unit=opt.discriminator_hidden_unit)
print(discriminator)



# ニューラルネットを用いた非線形モデルによる人工データを作成
mseed=0
n_unit1=16
n_unit2=16
sigma_=2
seed=10
Data = tsModel.NeuralNet(model_random_seed=mseed, p=opt.p, q=opt.q, n_unit=[n_unit1,n_unit2], sigma=sigma_, N=1400, random_seed=seed, return_net=False)
plt.figure(figsize=(13,8))
plt.plot(Data)
plt.title("model-random-seed:{0}, noise-random-seed:{1}, p:{2}, q:{3}, n_unit:{4}&{5}, sigma:{6}".format(mseed,seed,opt.p,opt.q,n_unit1,n_unit2,sigma_))
plt.xlabel("time")
plt.ylabel("value")
plt.savefig("output-images/Data_model-random-seed{0}_noise-random-seed{1}_p{2}_q{3}_n-unit{4}_{5}_sigma{6}.png".format(mseed,seed,opt.p,opt.q,n_unit1,n_unit2,sigma_))

# もしかしたら将来多次元の時系列を扱うことになるかもしれないので一応。
Data=Data.view(1,-1)

# trainとvalidationとtestに分割
trainData = Data[:,:1000]
valData = Data[:,1000:1200]
testData = Data[:,1200:]



# データの整形
trainMatrix = []
for i in range(trainData.shape[1]-(opt.p+1)):
    ans = trainData[:,i:i+opt.p+1].view(1,Data.shape[0],-1)
    trainMatrix.append(ans)
trainMatrix = torch.cat(trainMatrix)

valMatrix = []
for i in range(valData.shape[1]-(opt.p+1)):
    ans = valData[:,i:i+opt.p+1].view(1,Data.shape[0],-1)
    valMatrix.append(ans)
valMatrix = torch.cat(valMatrix)



def gradient_penalty(generated_data, real_data, gp_weight=10):

    batch_size = real_data.size()[0]

    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand_as(real_data)
    if cuda:
        alpha=alpha.to(device)

    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    if cuda:
        interpolated=interpolated.to(device)

    # Calculate probability of interpolated examples
    prob_interpolated = discriminator(interpolated)

    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated, grad_outputs=torch.ones(prob_interpolated.size()).to(device) if cuda else torch.ones(prob_interpolated.size()), create_graph=True, retain_graph=True)[0]

    gradients = gradients.view(batch_size, -1)# これいらないかも...
    
    # gradients_norm = (gradients.norm(2, dim=1) - 1) ** 2
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)# 0除算を防ぐ？

    return gp_weight * ((gradients_norm - 1) ** 2).mean()



# 学習

# Optimizers(パラメータに対して定義される)
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

# パラメータと学習データをGPUに乗っける
generator.to(device)
discriminator.to(device)
trainMatrix=trainMatrix.to(device)
valMatrix=valMatrix.to(device)

saveModel = input('学習中のモデルも全て保存しますか？(常に最終結果は保存します) （Yes：1, No：0）  ----> ')
saveModel = bool(int(saveModel))

import japanize_matplotlib
from scipy.stats import gaussian_kde

batches_done = 0
generator_done = 0# generatorを学習した回数

# グラフ描画用
loss_D_curve = []
loss_G_curve = []
p_value = []
corrcoef = []

# 訓練データの時系列のどの時刻を学習に用いるかをランダムにしているが、そのランダムシードを固定する
random.seed(a=opt.random_seed)

for epoch in range(opt.n_epochs):# epochごとの処理
    
    # trainMatrixの行をランダムにシャッフルする
    # r=torch.randperm(trainMatrix.shape[0])
    # c=torch.arange(trainMatrix.shape[1])
    # trainMatrix = trainMatrix[r[:, None],c]
    
    for i, batch in enumerate(range(0, trainMatrix.shape[0]-opt.batch_size, opt.batch_size)):# batchごとの処理
        
        # generatorへの入力を用意する
        X = trainMatrix[batch:batch+opt.batch_size]# torch.Size([64, 1, 8])
        # 時系列の順番はそのまま入力した方がいいのかな？
        rand=random.randint(0,trainMatrix.shape[0] - trainMatrix.shape[0]// opt.batch_size*opt.batch_size)
        X = trainMatrix[batch+rand : batch+rand+opt.batch_size]# torch.Size([64, 1, 8])
    
        X = Variable(X)# 自動微分？
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        for p in discriminator.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        if not opt.withGP:
            # discriminatorのパラメタをクリップする（全てのパラメタの絶対値がopt.clip_value以下の値になる）
            for idx, p in enumerate(discriminator.parameters()):
                if idx==0:
                    continue #  sigmaはクリップしない
                p.data.clamp_(-opt.clip_value, opt.clip_value)

        
        # 勾配情報を0に初期化する
        optimizer_D.zero_grad()        
        
        # 現在＋過去p時刻分の時系列をgeneratorで変換した値を取得
        hat_epsilon = generator(X)#.detach() # torch.Size([64, 4])
        # この「.detach()」はTensor型から勾配情報を抜いたものを取得する.(つまりこの後の誤差逆伝播のところではgeneratorのパラメタまで伝播しない)
        
        # generatorの出力と同じ大きさの標準正規分布からのサンプルを作成
        epsillon = Variable(torch.randn_like(hat_epsilon))
        
        # Adversarial loss すなわちWasserstein距離の符号を反転させたもの。（DiscriminatorはWasserstein距離を最大にする関数になりたい）
        if opt.withGP:
            loss_D = -torch.mean(discriminator(epsillon, is_from_generator=False)) + torch.mean(discriminator(hat_epsilon, is_from_generator=True)) + gradient_penalty(generated_data=hat_epsilon, real_data=epsillon, gp_weight=10)
        else:
            loss_D = -torch.mean(discriminator(epsillon, is_from_generator=False)) + torch.mean(discriminator(hat_epsilon, is_from_generator=True))
 

        # loss_Dを目的関数として微分をしてくれと言う合図
        loss_D.backward()
        # otimizerにしたがってパラメタを更新
        optimizer_D.step()

            
        # discriminatorをopt.n_critic回学習させるごとに一回generatorを学習させる(ただし最初はめっちゃdiscriminatorを優先させる)
        if batches_done % (100 if generator_done<25 or generator_done%500==0 else opt.n_critic) == 0:
            
            # -----------------
            #  Train Generator
            # -----------------
            
            for p in discriminator.parameters():
                p.requires_grad = False # to avoid computation
            
            # generatorの勾配情報を0に初期化
            optimizer_G.zero_grad()
            
            # 現在＋過去p時刻分の時系列をgeneratorで変換した値を取得
            hat_epsilon = generator(X)# torch.Size([64, 4]) (今度はdetachしない)
            
            # Adversarial loss(discriminatorの出力の期待値を大きくして、つまりWasserstein距離の第二項を大きくして、Wasserstein距離小さくしたい)
            loss_G = -torch.mean(discriminator(hat_epsilon, is_from_generator=True))
            
            # loss_Gを目的関数として微分をしてくれと言う合図
            loss_G.backward()
            optimizer_G.step()
            
            generator_done+=1

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, batches_done % trainMatrix.shape[0]//opt.batch_size, trainMatrix.shape[0]//opt.batch_size, loss_D.item(), loss_G.item()) )
            
        
        if batches_done % opt.sample_interval == 0:
            # もしここでhat_epsilon保存するなら保存する
            # hat_epsilon[:,0].shape
            # a=hat_epsilon
            # a=a.cpu().detach().numpy()
            # plt.hist(a[:,0])
            # plt.show()
            pass
        
        batches_done += 1

    loss_D_curve.append(loss_D.item())
    loss_G_curve.append(loss_G.item())
    
    # validationデータでgeneratorの出力の正規性検定のp-値と相互相関係数を確認する
    rnd = random.randint(0, valMatrix.shape[0]-opt.batch_size)
    valX = valMatrix[rnd:rnd+opt.batch_size]
    val_hat_epsilon = generator(valX)
    # p-値
    a=val_hat_epsilon[:,0].cpu().detach().numpy()
    p_value.append(stats.shapiro(a)[1])
    # 相互相関係数
    corrcoef.append(np.corrcoef(val_hat_epsilon.cpu().detach().numpy().T))

    if saveModel:
        if epoch%10000==0:
            torch.save(generator.state_dict(), 'parameters/generator_epoch{0}_{1}_batchSize{2}_randomSeed{3}_p{4}_q{5}_gSeed{6}_dSeed{7}_kernelSize{8}_numFilter{9}_dHiddenUnit{10}_withGP{11}.pth'
                       .format(epoch, opt.n_epochs, opt.batch_size, opt.random_seed, opt.p, opt.q, opt.generator_seed, opt.discriminator_seed, opt.generator_kernel_size,opt.n_filter,opt.discriminator_hidden_unit, str(opt.withGP)))
            torch.save(discriminator.state_dict(), 'parameters/discriminator_epoch{0}_{1}_batchSize{2}_randomSeed{3}_p{4}_q{5}_gSeed{6}_dSeed{7}_kernelSize{8}_numFilter{9}_dHiddenUnit{10}_withGP{11}.pth'
                       .format(epoch, opt.n_epochs, opt.batch_size, opt.random_seed, opt.p, opt.q, opt.generator_seed, opt.discriminator_seed, opt.generator_kernel_size,opt.n_filter,opt.discriminator_hidden_unit, str(opt.withGP)))

        if epoch%100000==0:
            print(opt.withGP)
            print("なぜだ〜〜〜！")
            kde = gaussian_kde(a)
            ls = np.linspace(min(a)-np.var(a), max(a)+np.var(a) , 100)
            plt.figure(figsize=(13,8))
            plt.title("generatorの出力の分布　\n　epoch:{0}/{1}, batchSize:{2}, randomSeed:{3}, p:{4}, q:{5}, gSeed:{6}, dSeed:{7}, kernelSize:{8}, numFilter:{9}, dHiddenUnit:{10}, withGP:{11}".format(epoch, opt.n_epochs, opt.batch_size, opt.random_seed, opt.p, opt.q, opt.generator_seed, opt.discriminator_seed, opt.generator_kernel_size,opt.n_filter,opt.discriminator_hidden_unit, str(opt.withGP)))
            plt.ylabel("密度")
            plt.plot(ls, kde.pdf(ls) , label="$\hat\epsilon$")
            plt.plot(ls, norm.pdf(ls), label="$\mathcal{N}(0,1)$")
            plt.legend()
            plt.savefig("output-images/density_epoch{0}_{1}_batchSize{2}_randomSeed{3}_p{4}_q{5}_gSeed{6}_dSeed{7}_kernelSize{8}_numFilter{9}_dHiddenUnit{10}_withGP{11}.png".format(epoch, opt.n_epochs, opt.batch_size, opt.random_seed, opt.p, opt.q, opt.generator_seed, opt.discriminator_seed, opt.generator_kernel_size,opt.n_filter,opt.discriminator_hidden_unit, str(opt.withGP)))
            plt.show()

    if epoch%1000==0 or epoch==opt.n_epochs-1:
        plt.figure(figsize=(13,8))
        plt.title("DiscriminatorのLossの遷移　\n　epoch:{1}, batchSize:{2}, randomSeed:{3}, p:{4}, q:{5}, gSeed:{6}, dSeed:{7}, kernelSize:{8}, numFilter:{9}, dHiddenUnit:{10}, withGP:{11}".format(epoch, opt.n_epochs, opt.batch_size, opt.random_seed, opt.p, opt.q, opt.generator_seed, opt.discriminator_seed, opt.generator_kernel_size,opt.n_filter,opt.discriminator_hidden_unit, str(opt.withGP)))
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.plot(loss_D_curve)
        plt.savefig("loss.png")
        plt.show()

torch.save(generator.state_dict(), 'parameters/generator_epoch{1}_batchSize{2}_randomSeed{3}_p{4}_q{5}_gSeed{6}_dSeed{7}_kernelSize{8}_numFilter{9}_dHiddenUnit{10}_withGP{11}.pth'.format(epoch, opt.n_epochs, opt.batch_size, opt.random_seed, opt.p, opt.q, opt.generator_seed, opt.discriminator_seed, opt.generator_kernel_size,opt.n_filter,opt.discriminator_hidden_unit, str(opt.withGP)))
torch.save(discriminator.state_dict(), 'parameters/discriminator_epoch{1}_batchSize{2}_randomSeed{3}_p{4}_q{5}_gSeed{6}_dSeed{7}_kernelSize{8}_numFilter{9}_dHiddenUnit{10}_withGP{11}.pth'.format(epoch, opt.n_epochs, opt.batch_size, opt.random_seed, opt.p, opt.q, opt.generator_seed, opt.discriminator_seed, opt.generator_kernel_size,opt.n_filter,opt.discriminator_hidden_unit, str(opt.withGP)))


# 結果の保存
# p-値
plt.figure(figsize=(13,8))
plt.plot(p_value)
plt.title("p-値の遷移　\n　epoch:{1}, batchSize:{2}, randomSeed:{3}, p:{4}, q:{5}, gSeed:{6}, dSeed:{7}, kernelSize:{8}, numFilter:{9}, dHiddenUnit:{10}, withPG:{11}".format(epoch, opt.n_epochs, opt.batch_size, opt.random_seed, opt.p, opt.q, opt.generator_seed, opt.discriminator_seed, opt.generator_kernel_size,opt.n_filter,opt.discriminator_hidden_unit, str(opt.withGP)))
plt.xlabel("epoch")
plt.ylabel("p-値")
plt.savefig("output-images/p-value_epoch{1}_batchSize{2}_randomSeed{3}_p{4}_q{5}_gSeed{6}_dSeed{7}_kernelSize{8}_numFilter{9}_dHiddenUnit{10}_withGP{11}.png".format(epoch, opt.n_epochs, opt.batch_size, opt.random_seed, opt.p, opt.q, opt.generator_seed, opt.discriminator_seed, opt.generator_kernel_size,opt.n_filter,opt.discriminator_hidden_unit, str(opt.withGP)))
plt.show()
# discriminatorのLoss
plt.figure(figsize=(13,8))
plt.title("DiscriminatorのLossの遷移　\n　epoch:{1}, batchSize:{2}, randomSeed:{3}, p:{4}, q:{5}, gSeed:{6}, dSeed:{7}, kernelSize:{8}, numFilter:{9}, dHiddenUnit:{10}, withPG:{11}".format(epoch, opt.n_epochs, opt.batch_size, opt.random_seed, opt.p, opt.q, opt.generator_seed, opt.discriminator_seed, opt.generator_kernel_size,opt.n_filter,opt.discriminator_hidden_unit, str(opt.withGP)))
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.plot(loss_D_curve)
plt.savefig("output-images/loss-D-curve_epoch{1}_batchSize{2}_randomSeed{3}_p{4}_q{5}_gSeed{6}_dSeed{7}_kernelSize{8}_numFilter{9}_dHiddenUnit{10}_withGP{11}.png".format(epoch, opt.n_epochs, opt.batch_size, opt.random_seed, opt.p, opt.q, opt.generator_seed, opt.discriminator_seed, opt.generator_kernel_size,opt.n_filter,opt.discriminator_hidden_unit, str(opt.withGP)))
plt.show()
# genetaratorのLoss
plt.figure(figsize=(13,8))
plt.title("GeneratorのLossの遷移　\n　epoch:{1}, batchSize:{2}, randomSeed:{3}, p:{4}, q:{5}, gSeed:{6}, dSeed:{7}, kernelSize:{8}, numFilter:{9}, dHiddenUnit:{10}, withPG:{11}".format(epoch, opt.n_epochs, opt.batch_size, opt.random_seed, opt.p, opt.q, opt.generator_seed, opt.discriminator_seed, opt.generator_kernel_size,opt.n_filter,opt.discriminator_hidden_unit, str(opt.withGP)))
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.plot(loss_G_curve)
plt.savefig("output-images/loss-G-curve_epoch{1}_batchSize{2}_randomSeed{3}_p{4}_q{5}_gSeed{6}_dSeed{7}_kernelSize{8}_numFilter{9}_dHiddenUnit{10}_withGP{11}.png".format(epoch, opt.n_epochs, opt.batch_size, opt.random_seed, opt.p, opt.q, opt.generator_seed, opt.discriminator_seed, opt.generator_kernel_size,opt.n_filter,opt.discriminator_hidden_unit, str(opt.withGP)))
plt.show()
# 相互相関係数
plt.figure(figsize=(13,8))
plt.title("相互相関係数の遷移　\n　epoch:{1}, batchSize:{2}, randomSeed:{3}, p:{4}, q:{5}, gSeed:{6}, dSeed:{7}, kernelSize:{8}, numFilter:{9}, dHiddenUnit:{10}, withPG:{11}".format(epoch, opt.n_epochs, opt.batch_size, opt.random_seed, opt.p, opt.q, opt.generator_seed, opt.discriminator_seed, opt.generator_kernel_size,opt.n_filter,opt.discriminator_hidden_unit, str(opt.withGP)))
plt.xlabel("epoch")
plt.ylabel("相関係数")
plt.plot(np.array(corrcoef)[:,1,0],label='1_2')
plt.plot(np.array(corrcoef)[:,2,0],label='1_3')
plt.plot(np.array(corrcoef)[:,3,0],label='1_4')
plt.plot(np.array(corrcoef)[:,2,1],label='2_3')
plt.plot(np.array(corrcoef)[:,3,1],label='2_4')
plt.plot(np.array(corrcoef)[:,3,2],label='3_4')
plt.legend()
plt.savefig("output-images/corrcoef_epoch{1}_batchSize{2}_randomSeed{3}_p{4}_q{5}_gSeed{6}_dSeed{7}_kernelSize{8}_numFilter{9}_dHiddenUnit{10}_withGP{11}.png".format(epoch, opt.n_epochs, opt.batch_size, opt.random_seed, opt.p, opt.q, opt.generator_seed, opt.discriminator_seed, opt.generator_kernel_size,opt.n_filter,opt.discriminator_hidden_unit, str(opt.withGP)))
plt.show()
# generatorの出力のカーネル密度推定結果
a=val_hat_epsilon[:,0].cpu().detach().numpy()
kde = gaussian_kde(a)
ls = np.linspace(min(a)-np.var(a), max(a)+np.var(a) , 100)
plt.figure(figsize=(13,8))
plt.title("generatorの出力の分布　\n　epoch:{0}/{1}, batchSize:{2}, randomSeed:{3}, p:{4}, q:{5}, gSeed:{6}, dSeed:{7}, kernelSize:{8}, numFilter:{9}, dHiddenUnit:{10}, withGP:{11}".format(epoch, opt.n_epochs, opt.batch_size, opt.random_seed, opt.p, opt.q, opt.generator_seed, opt.discriminator_seed, opt.generator_kernel_size,opt.n_filter,opt.discriminator_hidden_unit, str(opt.withGP)))
plt.ylabel("密度")
plt.plot(ls, kde.pdf(ls) , label="$\hat\epsilon$")
plt.plot(ls, norm.pdf(ls), label="$\mathcal{N}(0,1)$")
plt.legend()
plt.savefig("output-images/density_epoch{0}_{1}_batchSize{2}_randomSeed{3}_p{4}_q{5}_gSeed{6}_dSeed{7}_kernelSize{8}_numFilter{9}_dHiddenUnit{10}_withGP{11}.png".format(epoch, opt.n_epochs, opt.batch_size, opt.random_seed, opt.p, opt.q, opt.generator_seed, opt.discriminator_seed, opt.generator_kernel_size,opt.n_filter,opt.discriminator_hidden_unit, str(opt.withGP)))
