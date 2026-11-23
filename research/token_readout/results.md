# 逐 token 讀出與 Gradio 首輪驗收結果

同一版 Qwen3.5-2B 的逐 token 讀出已接入可貼正文的 Gradio。實際 L4 請求回傳全文原文標色、四條逐 token 曲線和可反查的前文候選。就預先指定的行為端點而言，凍結讀出改善了 Frick 的句級時序預測；目標效應、解答窗口機率和同疑問的早晚解開定位仍低於門檻。選凍結版作研究預覽候選。 [評分與逐家族明細](../results/token-readout-l4/metrics.json) · [實際 HTTP 預覽](../results/token-readout-l4/app/preview.html) · [候選收據](../results/token-readout-l4/candidate.json)

## 同一切分的三組對照

167 份刺激共 43,975 個原生 token。相同的讀出同時使用末層、第 12 層及六個原生 full-attention 層的可見前綴關係。逐句首次回報聚合句內 token，原實驗的回報點仍在句末。LoRA 只更新第 24 層 query 的輸出列與 K，輸出 gate 維持原值。兩組的兩項訓練損失均達到預先設定的八輪平台停訓判準。凍結第 40 輪停訓並封存第 36 輪，LoRA 第 38 輪停訓並封存同輪。擬合每輪各有 64 次 Frick 更新與 64 次證據更新。所有留出預測先由模型輸出並封存，再在本機連回評分標註。 [資料和規格](design.md) · [逐輪紀錄和封存](../results/token-readout-l4/fit/seal.json)

| 觀測 | 位置基線 | 凍結 Qwen | 末層 Q/K LoRA |
| --- | ---: | ---: | ---: |
| Frick 首次回報整篇 NLL，nats／試次，越低越好 | 1.738 | 1.493 | 1.551 |
| Frick 目標句首次回報差，百分點，真人 +55.39 | 0.00 | +19.89 | +19.05 |
| ExpMRC 已知問題證據區間 NLL，nats／題，越低越好 | 3.038 | 7.455 | 8.587 |
| ExpMRC 同一窗口辨別哪題較早有證據，27 對 | 50.0% | 63.0% | 59.3% |
| 早解與晚解皆在對應窗內分離，12 組，門檻 9 組 | 0 | 5 | 3 |
| 無解版經換題窗仍保留至少八成既有未解量，12 組 | 12 | 10 | 8 |

Frick 的整篇 NLL 以 384 次真實試次加權，目標句差幅則按四個故事等權。兩個模型在相同目標句上重現真人的前文效應方向，幅度仍低估約 35–36 個百分點。以故事等權，LoRA 減凍結的整篇 NLL 差為 +0.062，群組重抽樣的 95% 區間為 [+0.044, +0.092]。這四組是既有開發診斷材料。 [原始對照與逐故事分母](../results/token-readout-l4/metrics.json)

ExpMRC 的同窗兩題對比對應 17／27 組凍結、16／27 組 LoRA 排對方向。區間機率的誤差可分為窗前已過早觸發，以及在支持窗內沒有給足機率。相對位置基線，凍結在這兩部分分別增加 1.179 與 3.238 nats／題，LoRA 分別增加 1.618 與 3.931。兩個模型的窗內誤差佔較大部分，部分文章另有很強的提前觸發。這批標註定位的是指定問題的文字支持。 [54 題獨立分解](../results/token-readout-l4/evidence-timing-diagnostic.json)

新自編說明文的 12 組三聯對照在共同前綴上逐值一致。以凍結讀出為例，第 3 組在早、晚兩個解答窗內都和等長對照分開；第 1 組在兩個窗都沒有達到 5 個百分點的分離門檻。這兩組在實際上都各有清楚的解答句，模型的解開時序仍受材料影響。無解版的換題存續在 10 組達到八成參考線，步數基線 12 組都達到；單有存續不能辨別是否讀到了答句。三聯對照是預先封存的模型行為檢查，未另外蒐集真人反應。 [每組曲線與時點](../results/token-readout-l4/metrics.json)

## 同一權重的實際 Gradio 驗收

用比較後選定的同一凍結權重，執行 3 次實際 HTTP 分析請求，涵蓋一段前綴、其完整續文和含組合字元的 Unicode 文章。三篇各一次骨幹前向，6 次快取選取均無額外前向。HTTP 原文、所有已產生的標記、指向與 380 個曲線點逐項核對通過。共用的 24 個前綴 token 在四個通道最大絕對差為 5.22×10⁻⁸，前文指向無一變動。介面呈現的未解量是候選的預期未解質量，開啟與解除是該候選轉移量，轉換分數另由上一位置的未解量和當下表徵差構成。介面上的指向以已讀原文的候選位置呈現。 [HTTP 收據](../results/token-readout-l4/app/verification.json) · [前綴核對](../results/token-readout-l4/app/causality.json) · [實際原文與曲線](../results/token-readout-l4/app/preview.html)

訓練與 HTTP 驗收共用單篇、無 padding 的擷取路徑。規格所列的右側 padding 等價性未驗證，這版入口只支援無 padding。兩張 L4 均確認釋放，兩次配置至釋放的間隔約為 14 分 29 秒和 2 分 53 秒。擬合、預測與介面權重均有 SHA-256 收據，評分獨立重跑逐位元一致。逐 token 的解疑與中斷行為仍需更可靠的位置訊號和更接近正文自由閱讀的觀測。 [擬合 L4](../results/token-readout-l4/fit-run-verification.json) · [介面 L4](../results/token-readout-l4/app-run-verification.json)

在具備相同 Qwen 快照和 CUDA L4 的環境，可用已封存研究候選啟動實際介面。 [介面入口](app.py) · [同一候選權重](../results/token-readout-l4/fit/frozen.pt)

```sh
python -m research.token_readout.app \
  --runner research.token_readout.runner:load \
  --model /path/to/15852e8c16360a2fea060d615a32b45270f8a8fc \
  --checkpoint research/results/token-readout-l4/fit/frozen.pt
```
