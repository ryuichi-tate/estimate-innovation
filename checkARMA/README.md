# ARMA(p,q)($q\geq1$)のモデルで作成されたinnovationを推定できるかどうかを確かめる。

## 動機
そもそも$q=0$のモデルを仮定してinnovationを推定し変化点検知をすると言うのでは、ただの予測誤差に夜変化点検知と何にも変わらない。

やはり最終的にはinnovation自体も複雑に絡み合って$X_t$が生成される、そう言うモデルのinnovationを推定したい。

それが可能かどうかを確かめる。

## 実験概要
### まずは$ARMA(p,q\geq1)$モデルで確かめる。
- $ARMA(p,q\geq1)$に従うモデルを作成
- このモデルから作成された人工データから正確にinnovationが推定できるかどうか
- 推定するにはどのくらいの$\hat p$の値が必要なのか

評価はinnovationの推定値と正解との平均二乗誤差とする。

### 非線形でinnovation系列もバラバラに組み合わせたモデルを作成して確かめる。

## 実験詳細
`./tmp.ipynb`にて確認しながらつめる。<br>
→　今回の人工データを生成するモデル$F^{\ast}_{\phi}$は、

$$
X_t = \sum_{i=1}^4\phi_{ai}X_{t-i} + \varepsilon_t+\sum_{j=1}^2\phi_{bj}X_{t-j}\\
\phi_a = [1.0, -0.5, 0.7, -0.4],\ \ 
\phi_b =  [1.0, -0.7], \ \ \varepsilon_t\sim\mathcal{N}(0,1)
$$
という、$ARMA(4,2)$を用いる。

ここから作った時系列データの例<br>
![ARMAモデルのサンプル](./images/ARMA4-2sample.png "ARMA(4,2)の時系列データと元のinnovation系列")

対象実験として、$AR(\hat p)$を仮定してモデリングをした、その残渣系列のMSEをみる 

結果がこれ

![推定対象のARの次数を変化させた時のinnovationのmse](./images/ARMA4-2_inno-MSE_barplot.png "ARMA(4,2)の時系列データと元のinnovation系列")

次にwithoutDiscriminatorで学習させた場合にinnovationの評価がどうなるのかを確かめる。
学習ファイルはlearn_ver1.pyかな。