# GPUが使えるかどうかを、pytorchのライブラリを用いてcheckする。
# True or Falseを返す


import torch
import os

def checkGPU():
    cuda = True if torch.cuda.is_available() else False

    if cuda:
        print("GPUが使えます。")
        use_gpu = input('GPUを使いますか？ （Yes：1, No：0）： ')
        cuda = bool(int(use_gpu))
    else:
        print("GPUは使えません。")
    if cuda:
        gpu_id = input('使用するGPUの番号を入れてください : ')
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device('cuda:'+gpu_id if cuda else 'cpu')

    return cuda, device