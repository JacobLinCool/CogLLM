# 可換骨幹的逐 token 讀出

凍結的 Qwen3.5-2B、LFM3-1.7B-Base 和 Gemma 4 E2B 均在 Colab L4 各對共同的 191 份正文執行一次因果前向。用相同端點擬合後，Qwen 的跨度版在 Frick 整篇首次回報及 ExpMRC 證據區間的留出 NLL 最低。加入 IIRC 後，Qwen 校準版在單一疑問的早解與晚解同時定位由 1／12 增至 6／12，低於預定的 9／12。雙疑問對照的 8 組仍無一同時定位並選對兩個疑問。 [三骨幹評分](../results/portable-token-readout-l4/metrics.json) · [補充評分](../results/portable-token-readout-l4/linked-metrics.json) · [同家族配對差](../results/portable-token-readout-l4/supplemental-comparison.json)

## 留出結果

下表各骨幹使用跨度讀出加開發集校準。Frick NLL 為 384 次真人試次加權的整篇首次回報負對數概似，位置基線為 1.785 nats／試次。ExpMRC NLL 為 27 篇文章的人工標記區間概似，模型都獲得原題。按各分詞器曝光量建立的位置基線為 3.041–3.044 nats／題。兩項 NLL 均以低為優。

| 骨幹與訓練資料 | Frick NLL | Frick 目標句回報差，真人 +55.39 個百分點 | ExpMRC NLL | 早解與晚解均命中 | A／B 兩個方向都命中 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-2B | 1.295 | +22.77 | 2.743 | 1／12 | 0／8 |
| LFM3-1.7B-Base | 1.438 | +9.25 | 3.018 | 0／12 | 0／8 |
| Gemma 4 E2B | 1.774 | +0.02 | 2.921 | 0／12 | 0／8 |
| Qwen3.5-2B 加 IIRC | 1.301 | +20.92 | 2.772 | 6／12 | 0／8 |

Qwen 在無校準的同骨幹比較中，跨度版將首次回報 NLL 由 1.372 降至 1.295，證據 NLL 由 2.849 降至 2.680。LFM3 的跨度版把首次回報 NLL 由 1.461 降至 1.438，但證據 NLL 由 2.961 升至 3.068。Gemma 的跨度版把未校準證據 NLL 由 3.676 降至 3.256，再經校準降至 2.921。Gemma 校準版的排序為 19／27，固定猜第一題為 17／27，兩題序分層等權後為 60.0%。三個骨幹都低估真人目標句差幅。 [逐端點與配對區間](../results/portable-token-readout-l4/metrics.json)

IIRC 加上開發集校準後，比同骨幹、同校準階段多命中五個早解／晚解家族。差幅是 +41.7 個百分點，12 組配對重抽樣區間為 [+16.7, +66.7] 個百分點，僅描述這組自編材料。ExpMRC 的 NLL 差為 +0.029 nats／題，區間 [−0.158, +0.225]。四篇 Frick 家族等權差約為零。IIRC 的點版在 16 個 A／B 線索中有 1 個通過個別檢查，所有版本的完整雙向家族通過數為零。A／B 評分已給定上文錨點，因此仍欠缺從正文自動找到疑問與追蹤同一疑問的證據。 [補充實驗配對結果](../results/portable-token-readout-l4/supplemental-comparison.json)

正式標註接回前，按開發集等權均值選出加入 IIRC 的 Qwen 跨度校準版。它的均值由不含 IIRC 的 1.811 降至 1.785 nats。這個選擇對應開發目標，正式留出的最佳候選因端點而異。 [選擇收據](../results/portable-token-readout-l4/candidate.json)

## 實作與資料

同一讀出接收正文中各 token 的中層與末層狀態、原文字元範圍，以及指定使用時的原生全域注意力摘要。各原生全域層在前向中以真實遮罩和縮放擷取所有可見前文位置的注意力，再按頭與層計算平均和峰值。LFM3 的卷積及滑窗層、Gemma 的滑窗層照原模型前向，訊號進入隱狀態。跨度版以錨點末端往前最多四個 token 組成候選，與任何較晚位置配對。各凍結骨幹各自重訓讀出，增加約 20.3–21.3 萬個可訓練參數。 [擷取器](backbone.py) · [讀出頭](readout.py)

共同封包保留 167 份既有文本，並加上 8 組雙疑問的 24 份對照。Frick 切成 12／4／4 個訓練、開發、測試家族，ExpMRC 為 48／16／27 篇。三個分詞器以共同原文字元停點或區間對齊。Frick 的 584 個停點只越過原有空白，沒有消耗下一句內容。全部原資料保留。雙疑問試驗固定兩個候選上文錨點，模型只讀正文。保留原來 12 組早解、晚解、未解對照，兩套工程診斷都沒有新增真人觀測。 [固定規格](transfer-protocol.md) · [資料封存](../results/portable-token-readout-l4/qwen/dataset-receipt.json)

開發集等權 Frick 與 ExpMRC NLL 決定停止、候選及補充實驗骨幹。點版和跨度版在同一資料上各自擬合。跨度校準版事後以開發集擬合開啟與解除的 logit 增益、解除偏置。ExpMRC 的排序在 27 篇中有 17 篇的較早答案對應第一個顯示問題，因此同時保留固定猜第一題的 63.0% 與依題序等權的排序結果。 [校準與擬合](transfer_fit.py) · [同字元評分](transfer_score.py)

Qwen 補充實驗另外使用 160 份 IIRC 官方訓練段落。每份只保留唯一來源錨點與後續同段支持區間，正文沒有插入題目。附加區間損失權重為 0.5，原有兩端點每輪各 48 次更新。新材料與原 191 份資料沒有重複的連續十二詞窗。三個候選來源內雙錨點配對未通過錨點唯一性或詞元邊界，補充封包因此有 160 個單錨點區間和零個雙錨點配對。選定區間是標註者事後判定的文字支持。 [補充規格](transfer-iirc-protocol.md) · [來源與篩選細節](../results/portable-token-readout-l4/linked/linked-manifest.json) · [IIRC 原研究](https://aclanthology.org/2020.emnlp-main.86/)

## 執行驗證

三個骨幹的原生前綴擷取均通過截短與替換未讀後文的檢查。Gemma 首次執行因截短後的隱向量最大絕對差被拒。重跑沿用前次三骨幹試驗的逐 token 相對 RMS 上限 0.001，最大為 0.000615。原生注意力摘要仍採最大絕對差上限 0.001，最大為 0.000754。重跑保留原快照及擷取運算。四個擬合工作階段的點版和跨度版都在第 10 輪達到預設的開發停訓。兩份最終評分重新執行後逐位元一致。 [Gemma 重驗](../results/portable-token-readout-l4/gemma-gate-recovery.json) · [開發停訓範例](../results/portable-token-readout-l4/linked/fitting.json)

選定權重以三次實際 Gradio HTTP 請求驗收，共 90 個 token、360 個曲線點、三次骨幹前向。六次點選沿用已擷取資料。25 個共同前綴 token 的四條曲線最大絕對差為 4.48×10⁻⁸，沒有上文指向變動。三個基礎實驗、IIRC 補充、介面驗收及最初 Gemma 工作階段均已確認釋放。 [實際輸出預覽](../results/portable-token-readout-l4/app/preview.html) · [HTTP 驗證](../results/portable-token-readout-l4/app/verification.json) · [前綴核對](../results/portable-token-readout-l4/app/causality.json) · [介面工作階段釋放](../results/portable-token-readout-l4/app/run-verification.json)

## 重現與分析入口

以[候選收據](../results/portable-token-readout-l4/candidate.json)的權重路徑及對應的[固定快照](../backbone_transfer/models.json)啟動。入口從候選權重識別骨幹、讀出種類與校準值，並以載入的快照核對 revision。現行固定環境使用 Colab L4。

```sh
python -m research.token_readout.app \
  --runner research.token_readout.portable_runner:load \
  --model /path/to/pinned-model-snapshot \
  --checkpoint /path/from/candidate.json
```

跨模型的資料準備、訓練與評分入口為 `research.token_readout.transfer_data`、`research.token_readout.transfer_colab` 和 `research.token_readout.transfer_score`。補充材料由 `research.token_readout.linked_supplement` 重建，以 `transfer_colab prepare --experiment linked` 另行封存。各執行資料夾保留每輪損失、候選、正式預測、前綴測試和釋放收據。 [基準評分](../results/portable-token-readout-l4/metrics.json) · [IIRC 補充評分](../results/portable-token-readout-l4/linked-metrics.json)
