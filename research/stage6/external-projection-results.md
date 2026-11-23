# 跨體裁與跨語言留出結果

同一個 Gradio 候選完成 GENTLE 跨段前文定位及土耳其語等詞數句對驗證。兩組的主要效果估計都朝改善方向，預先指定的 95% 區間均跨過零。 [固定規格](external-projection-protocol.md) · [模型與 L4 執行收據](../results/external-projection-l4/run-verification.json)

| 留出端點 | 字面或表面基線 NLL | 加入 Qwen 或 App 分數的 NLL | 配對差與 95% 區間 |
| --- | ---: | ---: | --- |
| GENTLE 的人工前文位置 | 4.4232 | 4.2622 | −0.1609 [−0.5906, +0.2739] |
| 土耳其語真人較易句選擇 | 0.68658 | 0.68608 | −0.00050 [−0.00294, +0.00228] |

NLL 單位為 nats，越低代表給人工標註或所選句子較高機率。

## GENTLE 的跨段定位

沿用 GUMBridge 原先封存的距離、字面、字面加 Qwen 三組讀出。GENTLE 官方測試的 26 篇文章中，14 篇有 34 筆標註可對回不同自然段，分詞與詞末條件排除 3 筆，留下 **13 篇、31 筆**。每筆有中位數 188 個可評詞末，包含當前段內的詞。最終先驗篇數門檻已達到，文章配對區間未通過。其中三組第一名命中都為零。事後按相同文章權重重算前 5 名命中，字面為 2.6%，加入 Qwen 為 15.4%。可用於設計更準的來源候選，當前證據仍不足以把模型排序當成可靠的定位。 [來源稽核](../results/gum-bridge-gentle-audit.json) · [固定讀出留出](../results/external-projection-l4/gentle.json) · [前 5 名重算](../results/external-projection-l4/local-verification.json)

端點從人工指定的橋接片語末端，評估已讀文章中的共同參照位置。它涵蓋詞末排名，前文片語的首次疑問、主觀努力和察覺解除的時間沒有相應的逐次讀者觀測。 [GUMBridge 標註方式](https://aclanthology.org/2026.lrec-1.543.pdf#page=4)

## 土耳其語的等詞數句對

用 App 固定句級讀出的「閱讀容易度」，將兩句分數差轉成較早呈現句被選為較容易的機率。原 App 分詞與分句可還原 779 句中的 778 句。只用擬合材料校準，留出有 **616 位讀者、700 筆選擇、393 個句對、9 個連通元件**。切分兩端的原句與絕對詞數層級各自分離，來源紀錄有 502 位讀者跨兩組。按作者品質篩選保留的 554 筆留出選擇，表面基線和加入 App 分數的 NLL 分別為 0.68473 與 0.68407，方向與主要比較一致。 [固定資料與切分](../results/external-projection-l4/turkish-assay.json) · [校準及留出](../results/external-projection-l4/turkish-scores.json)

只用 App 分數的 NLL 為 0.68303，呈現順序基線為 0.69260，差值的事後區間為 [−0.02631, +0.00637] nats。等詞數是作者的計算。表面基線也控制 App 看到的原文字元數和空白切分數，兩句的標點差異仍可反映在這些計數裡。此結果將可用範圍限制在目前測過的語言與端點，土耳其語閱讀容易度仍待獨立支持。 [獨立重算](../results/external-projection-l4/local-verification.json) · [作者公開欄位](https://zenodo.org/records/22775520/files/README.md?download=1)

## 共用執行與重現

首次 Colab 啟動未取得工作程序確認，該 L4 已釋放。延長啟動確認時間後，以逐位元相同的測試封包和遠端工作程式完成。805 份原文共 46,501 個 Qwen token、18,199 個經認證的可讀停點。GPU 對每份原文一次前向，另做 12 項因果控制，全部通過。前向與讀出約 23.9 秒，擷取階段約 70.9 秒，GPU 峰值保留 8.45 GiB。兩次配置均已確認釋放。CPU 重播與獨立公式核對一致，重播合計約 1.31 秒，程序峰值約 168 MiB。 [GPU 摘要](../results/external-projection-l4/extraction.json) · [本機重播](../results/external-projection-l4/local-verification.json) · [兩次釋放紀錄](../results/external-projection-l4/run-verification.json)
