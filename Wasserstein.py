import numpy as np
import torch
import math
from scipy.stats import norm  # 正規分布


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

def combinations_count(n, r, device=torch.device(type='cpu')):
    return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))
def integral(a, b, c, p, device=torch.device(type='cpu')):
    ret=0
    if a>=c:
        for i in range(p+1):
            ret += combinations_count(p, i, device=device)*(-c)**i/np.sqrt(2*np.pi)*torchJn(n=p-i, a=a, b=b)
    elif a<c and c<b:
        ret = integral(a=a, b=c, c=c, p=p, device=device) + integral(a=c, b=b, c=c, p=p, device=device)
    else:
        for i in range(p+1):
            ret += combinations_count(p, i, device=device)*c**i*(-1)**(p-i)/np.sqrt(2*np.pi)*torchJn(n=p-i, a=a, b=b)
    return ret.to(device)

def pWasserstein(x, p, device=torch.device(type='cpu')):
    N=x.shape[0]
    ret = 0
    for n in range(1,N+1):
        ret += integral(a=torch.tensor(norm.ppf(q=(n-1)/N, loc=0, scale=1)).to(device), b=torch.tensor(norm.ppf(q=n/N, loc=0, scale=1)).to(device),c=x[n-1],p=p, device=device)
    return ret**(1/p)