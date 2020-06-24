# triple

<img src="../../images/モデルの全体像.jpeg" alt="モデルの全体像" title="モデル全体像" width=100%>

これら$G_{\theta}$と$D_{w}$と$F_{\phi}$を同時に学習させる。

tipleでは、generatorの学習とpredictorの学習を全く同じスケジュールで行った。
結果、predictorはびっくりするくらい過学習したし、パラメータも全然正解を予測してない。

これをどうにかしなきゃね。