import numpy as np
import torch

class DataSet:
    def __init__(self, tsData, hat_p, device):
        """
        tsData：時系列データ（今回は一次元の時系列を想定）
        hat_p　：想定する次数
        initで入力データと出力データを作成する。
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
        """
        self.device = device
        self.tsData =  torch.tensor(tsData, dtype=torch.float) # torch.tensorに変換
        self.hat_p = hat_p
        self.X, self.Y = self.make_XY(x=self.tsData, p=self.hat_p)
    
    def  __len__(self):
        return len(self.X) # データ数を返す
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def make_XY(self, x:torch.tensor, p:int):
        X = [x[i:i+p] for i in range(len(x)-p)]
        Y = [x[i+p] for i in range(len(x)-p)]
        return X, Y
    

def make_XY_forGandF(x:np.ndarray, p:int):
    """
    一次元の時系列xを入力された時、過去p時刻の値（入力）と現在時刻の値（出力）のペアを出力する関数
    """
    x = torch.tensor(x, dtype=torch.float)
    X = [x[i:i+p+1] for i in range(len(x)-p)]
    Y = [x[i+p] for i in range(len(x)-p)]
    return X, Y

class DataSet_forGandF:
    def __init__(self, tsData, p):
        """
        tsData：時系列データ（今回は一次元の時系列を想定）
        p　：想定する次数
        initで入力データと出力データを作成する。
        """
        self.tsData =  torch.tensor(tsData, dtype=torch.float) # torch.tensorに変換
        self.p = p
        self.X, self.Y = make_XY_forGandF(self.tsData, self.p)
    
    def  __len__(self):
        return len(self.X) # データ数を返す
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]