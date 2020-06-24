そもそも学習、というか誤差逆伝播法を使ってユールウォーカー方程式が解けるのかどうかを見ます

その前にユールウォーカー法でどのくらい精度良くできるのかを試します

change_seed/change_seed.ipynbで$F_{\phi}$単体の学習を行っています（ニューラル）<br>
結果はcheckYule-Walker.ipynbで。パラメタの分布の確認をしている

change_seed_ver2として、真のモデルのパラメータが$\phi^{\ast}=[0.3, -0.4, 0.2, -0.5, 0.6, -0.1, 0.1]$, $\sigma^{\ast}=2$であるデータを学習させてみる。

innovation_estimation_distribution.ipynbは湯ールウォーカー法によって学習された推定モデルがinnovationの分布をどこまで正確に当てられているのかを調査しています。
