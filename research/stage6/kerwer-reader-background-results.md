# Kerwer 英文摘要的讀者背景留出結果

沿用現行 Gradio checkpoint 的英文摘要分數，在原研究的 166 位讀者、12 個 DOI 家族及 1,947 筆試次上，完成 20 個交叉留出格。每格擬合與測試均無讀者或 DOI 重疊。原始評分及讀者對照留在本機。Colab 使用各訓練格的評分類別計數及無評分的留出特徵，重現逐次八級比例勝算的似然。擬合階段耗時 3.0 秒，L4 已釋放。 [事先封存的規格](kerwer-reader-background-protocol.md) · [擬合與釋放收據](../results/kerwer-reader-background-l4/run-verification.json) · [完整留出評分](../results/kerwer-reader-background-l4/scores.json)

| 主要對照 | RPS 改善量（正值較好） | 95% 讀者 × DOI 配對區間 |
| --- | ---: | ---: |
| 文字讀出加入讀前 E‑PA | +0.00035 | [−0.00125, +0.00201] |
| 長度、版本、E‑PA 加入 Gradio 理解分數 | +0.00545 | [−0.00306, +0.01374] |
| 文字與 E‑PA 主效應加入交互項 | +0.00020 | [−0.00035, +0.00072] |

加法與交互背景讀出均通過預定的累積校準非劣門檻，三項 RPS 改善區間均跨零。空白背景透過訓練格的 E‑PA 分布求邊際平均，兩種空白讀出相對文字讀出都通過 RPS 與累積校準的非劣門檻。逐一留除 12 個 DOI 的敏感度結果已隨報告封存。

這個結果未證實讀前 E‑PA 能改善未見英文摘要與未見讀者上的逐次評分分布。實驗沿用原研究與既有改寫驗證的讀者及摘要家族，檢驗尺度是閱讀後的整篇自評理解。這組輸出保留作後續讀者背景比較的基線。 [原研究 Methods、Table 6](https://doi.org/10.1525/collabra.18898) · [資料來源與折號收據](../results/kerwer-reader-background-preparation.json)
