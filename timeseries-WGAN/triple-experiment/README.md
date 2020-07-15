# ここはtripleの学習において、いろいろな試行錯誤を行う場所である。

## learn_triple_ver1.py
では、generatorと同じく、
- 基本predictorはdiscriminator5回につき1回
- ただし、predictorを25回backpropするまでは100回に1回
- あと、predictorを500回backpropするごとに100回連続でdiscriminatorを回す

の方針でやった。この結果、![PredictorのLossの遷移](../triple/output-images/p7/floss_epoch2000_batchSize64_GP1_Corr1_DataSeed0.png)
と、どう考えても過学習した。

とりあえずgeneratorやdiscriminatorが十分学習をする前に予測誤差を減らすような真似をしてはいけない感じだったので、もっとpredictorの学習の開始を遅らせてみる。

## learn_triple_ver2.py

- 基本predictorはdiscriminator5回につき1回
- ただし、predictorを25回backpropするまでは1000回に1回
- あと、predictorを500回backpropするごとに100回連続でdiscriminatorを回す

と、最初1000回に1回に変えた

うまくいかず...

![PredictorのLossの遷移その2](./output-images/p7/floss_epoch2000_batchSize64_GP1_Corr1_DataSeed0.png)

実験番号をメモろう。

## learn_triple_ver3.py
- 基本predictorはdiscriminator5回につき1回
- ただし、predictorを25回backpropするまでは1000回に1回
- あと、predictorを500回backpropするごとに100回連続でdiscriminatorを回す
 加えて
 - Fの最適化手法もRSMprop
 - Corrのwightは0
 でやってみる

ん〜なんだこれ
![Predictorの遷移その3](./output-images/p7/No3_floss_epoch2000_batchSize64_GP1_Corr0_DataSeed0.png)

epochをめっちゃ増やす
Fを4万回回せるようにするには...?
Dを20万回回すしかねぇ


## learn_triple_ver4.py

$F_{\phi}$だけ事前に学習させ、sigmaの重みのパラメータを1にしてtripleを学習させるのはどうか。

ここでの事前学習は、
- innovationは正規分布からの乱数
- epochは1000回
- RMSprop
でFを学習させることをさす

### 結果
とりあえず事前学習はうまく行った。
が、やはりGANを学習させ始めると過学習？
![Predictorの遷移その4](./output-images/p7/No4_floss_epoch2000_batchSize64_GP1_Corr0_DataSeed0.png)

事前学習では正解できてた係数の推定も、GANを学習させ終わった後は散々だった

## learn_triple_ver5.py

学習プランを変更して、Fの過学習を防ぐ
とりあえずFの学習を1000回に一回にする

(100epochごとにモデルを保存した方がいいのかもしれない...)

![Predictorの遷移その5](./output-images/p7/No5_floss_epoch2000_batchSize64_GP1_Corr0_DataSeed0.png)

う〜ん、一応Fのパラメータが極端に悪くなることは無くなったけど、$\sigma$の値とかひどいな...

## learn_triple_ver6.py

$F$の学習をAdamにしてみるか。
全然ダメっすね
うまくいかない
もうこれを説明して教えをこうか

それともDiscriminatorではなく、正規分布のWasserstein距離を用いるか....

### 正規分布のWasserstein距離

[最適輸送理論概論](http://www.kurims.kyoto-u.ac.jp/~kyodo/kokyuroku/contents/pdf/1916-11.pdf)
によると、二つの正規分布$\mathcal{N}(\mu_1,\Sigma_1)$と$\mathcal{N}(\mu_2,\Sigma_2)$の距離は

$W_2(\mathcal{N}(\mu_1,\Sigma_1), \mathcal{N}(\mu_2,\Sigma_2)) =|\mu_1-\mu_2|^2+\mathrm{Tr}(\Sigma_1)+\mathrm{Tr}(\Sigma_2)-2\mathrm{Tr}\{(\Sigma_2^{\frac{1}{2}}\Sigma_1\Sigma_2^{\frac{1}{2}})^{\frac{1}{2}}\}$

となるらしい。WGANで使うWasserstein距離は$W_1$だけど、まあいいか...？

いやこんなの見つけた<br>
[Sliced Wasserstein GMM を実装してみた](https://yokaze.github.io/2019/09/12/)

マジかちょっと計算してみようかしらん