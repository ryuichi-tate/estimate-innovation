# ここではWasserstein距離の計算をDiscriminatorにさせるのではなく、理論計算した値を用いる

[Sliced Wasserstein Distance for Learning Gaussian Mixture Models](https://arxiv.org/pdf/1711.05376.pdf)
を見たら、一次元の確率分布間のp-Wasserstein距離は

$W_p(\rho,\nu)=(\int_0^1d^p(F_{\rho}^{-1}(z)-F_{\nu}^{-1}(z))dz)^{\frac{1}{p}}$

でおkって書いてあった。

距離はユークリッド距離で、$d^p(x,y)=|x-y|^d$

$F_{\rho}(x)$は累積分布関数


Generatorの出力$\{x_n\}_{n=1}^N$の経験分布と連続確率分布$Q$とのp-Wasserstein距離をゴリゴリ計算すると、

$W_p=\left\{\sum_{n=1}^N\int_{\Phi^{-1}(\frac{n-1}{N})}^{\Phi^{-1}(\frac{n}{N})}|x_n-x|^pf_Q(x)dx\right\}^{\frac{1}{p}}$

$Q$を標準正規分布にしてLossを設計する

## Gaussian-Integral.ipynb
p-Wasserstein距離の計算の実装があってるかどうかを確かめるとこ。

## Wasserstein.py
入力値の経験分布と標準正規分布とのp-Wasserstein距離を導出する関数が置いてある。

- <code>torchJn(n, a, b)</code>：<br>
$J_n(a,b)=\int_a^bx^ne^{-\frac{x^2}{2}}dx$<br>
を計算する。<code>a,b</code>は<code>torch.tensor()</code>、<code>n</code>は非負の整数。

- <code>combinations_count(n, r)</code>：<br>
$_n{C}_r$<br>
を計算する。<code>n,r</code>は非負の整数。（これをガンマ関数にしてあげれば実数に拡張できるが今はしてないので、p-Wasserstein距離の$p$は自然数のみになっている。）

- <code>integral(a, b, c, p)</code>:<br> 
$\frac{1}{\sqrt{2\pi}}\int_a^b|x-c|^pe^{-\frac{x^2}{2}}dx$<br>
を計算する。<code>a,b,c</code>は<code>torch.tensor()</code>、<code>p</code>は非負の整数。

- <code>pWasserstein(x, p)</code>:<br>
$W_p=\left\{\sum_{n=1}^N\int_{\Phi^{-1}(\frac{n-1}{N})}^{\Phi^{-1}(\frac{n}{N})}|x_n-x|^p\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}dx\right\}^{\frac{1}{p}}$<br>
を計算する。<code>x</code>は<code>torch.tensor()</code>、<code>p</code>は非負の整数。

## learn_withoutDiscriminator.ipynb
学習がしっかりできるかどうか最初にcheckしたnotebook


## 実験0

## 実験1
とりあえず回す！

## 実験2
事前学習にzero.grad()入れてなかったのでは。
入れてやってみよー

## 実験3

マルチタスクラーニングにする