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
