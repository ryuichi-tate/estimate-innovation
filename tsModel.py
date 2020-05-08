# 人工データ作成関数
# tsModel.pyとtimesries-WGAN/tsModel.pyは内容を共有しています。
import numpy as np

def ARIMA(a=[0], b=[0], d =None, mu=0, sigma=1,N=1000, random_seed=0, burn_in=None, randomness=True):
    # 乱数の初期化
    np.random.seed(random_seed)
    
    # 係数をnumpy.ndarrayに変えておく
    a = np.array(a)
    b = np.array(b)
    
    # 次数の取得
    p =  0 if (a == np.array([0])).prod() else len(a)
    q =  0 if (b == np.array([0])).prod() else len(b)
    
    # ARMAかARIMAか判定
    if d==None:
        ARIMA_flg=False
        d=0
    else:
        ARIMA_flg=True
    
    # burn-in期間の設定
    if burn_in==None:
        burn_in = 100*max(p, q, d)
    
    if randomness:
        random = np.random.normal(loc=mu, scale=sigma, size=N+burn_in+max(p, q, d))
    else:
        random = np.zeros(shape=(N+burn_in+max(p, q, d)))
    
    # 初期値は0
    ts = np.zeros_like(random)
    
    for i in range(max(p, q, d), N+burn_in+max(p, q, d)):
        ts[i] = (a*np.flip(ts[i-p:i])).sum() + (b*np.flip(random[i-q:i])).sum() + random[i]
    
    if ARIMA_flg:
        for _ in range(d):
            for i in range(max(p, q, d), N+burn_in+max(p, q, d)):
                ts[i] = ts[i] + ts[i-1]
            
    return ts[burn_in+max(p, q, d):]

def SARIMA(a=[0], b=[0], d =None, phi=[0], theta=[0], D=None, m=0, mu=0, sigma=1, N=1000, random_seed=0, burn_in=None, randomness=True):
    # 乱数の初期化
    np.random.seed(random_seed)
    
    # 係数をnumpy.ndarrayに変えておく
    a = np.array(a)
    b = np.array(b)
    phi  = np.array(phi)
    theta = np.array(theta)
    
    # 次数の取得
    p =  0 if (a == np.array([0])).prod() else len(a)
    q =  0 if (b == np.array([0])).prod() else len(b)
    P = 0 if (phi == np.array([0])).prod() else len(phi)
    Q = 0 if (theta == np.array([0])).prod() else len(theta)
    
    
    # burn-in期間の設定
    margin = max(p, q, (0 if d==None else d), m*P, m*Q, m*(0 if D==None else D))
    if burn_in==None:
        burn_in = 100*margin
        
    # 乱数epsilonの作成
    if randomness:
        random = np.random.normal(loc=mu, scale=sigma, size=N+burn_in+margin)
    else:
        random = np.zeros(shape=(N+burn_in+margin))
        
    # そもそも季節成分あるのか?
    if m==0:
        return ARIMA(a=a, b=b, d=d, mu=mu, sigma=sigma, N=N, random_seed=random_seed, burn_in=burn_in, randomness=randomness)
    
    # 初期値は0
    ts = np.zeros_like(random)
    u = np.zeros_like(random)
        
    # 季節成分についてARIMAを構成する
    for i in range(margin, N+burn_in+margin):
        u[i] = (phi*np.flip(u[i-m*P:i:m])).sum() + (theta*np.flip(random[i-m*Q:i:m])).sum() + random[i]
        
    # ARMAかARIMAか判定
    if D==None:
        SARIMA_flg=False
    else:
        SARIMA_flg=True
    
    if SARIMA_flg:
        for _ in range(D):
            for i in range(margin, N+burn_in+margin):
                u[i] = u[i] + u[i-m]
    
    
    # 次に普通にARIMAを構成する
    for i in range(margin, N+burn_in+margin):
        ts[i] = (a*np.flip(ts[i-p:i])).sum() + (b*np.flip(u[i-q:i])).sum() + u[i]
        
    # ARMAかARIMAか判定
    if d==None:
        ARIMA_flg=False
    else:
        ARIMA_flg=True    
    
    if ARIMA_flg:
        for _ in range(d):
            for i in range(margin, N+burn_in+margin):
                ts[i] = ts[i] + ts[i-1]
    
    return ts[burn_in+margin:]

# 次はニューラルネットを用いた人工データ作成

import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self, p, q, n_unit1, n_unit2):
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

def NeuralNet(model_random_seed=0, p=7, q=3, n_unit=[16,16], sigma=1, N=1000,random_seed=0, burn_in=None, randomness=True, return_net=False):
    # 乱数の初期化
    torch.manual_seed(model_random_seed)
    np.random.seed(random_seed)
    # インスタンスの作成
    net = Net(p=p, q=q, n_unit1=n_unit[0], n_unit2=n_unit[1])

    # burn-in期間の設定
    margin = max(p, q)
    if burn_in==None:
        burn_in = 100*margin

    # 乱数epsilonの作成
    if randomness:
        random = np.random.normal(loc=0, scale=sigma, size=N+burn_in+margin)
        random = torch.tensor(random, dtype=torch.float).view(1,-1)
    else:
        random = torch.zeros([1, N+burn_in+margin])

    # 初期値は0
    ts = torch.zeros_like(random)

    for i in range(margin, N+burn_in+margin):
        net_input = torch.cat((random[:,i-q:i], ts[:,i-p:i+1]), dim=1)
        output = net(net_input)
        ts[0][i] = float(output)
    
    if not return_net:
        return ts[0][burn_in+margin:]
    else:
        return net
