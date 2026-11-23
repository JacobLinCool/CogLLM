# IIRC 同段稍後支持文字的實驗

給定原始段落及段內一個已知錨點，估計作答者所選的支持文字何時在同一段落中完整出現。IIRC 由提問者選出連結，再由另一批作答者標出支持文字。這個實驗以作者發布的 `main` 證據位置當作量測端點。提問者首次產生疑問或察覺已解答的時刻沒有被記錄。 [IIRC §§2.2–2.3](https://aclanthology.org/2020.emnlp-main.86.pdf#page=4) · [作者檔獨立重建](results/iirc-seed-later-audit.json)

## 資料與前綴

保留單一可定位連結和一段可逐字對回原始段落、且位於錨點之後的支持文字。推論條件包含原始段落純文字、提供的錨點座標、當下停點及每條路徑的識別碼。前向只輸入原文。讀出以座標索引狀態，識別碼僅用於資料對齊。問題內容、答案、連結標題與證據位置均未進入推論。未來若串接純正文錨點偵測，需要另計其漏報及配對錯誤。 [資料準備](semantic_tracking/iirc_seed_later_assay.py) · [納入規則](results/iirc-seed-evidence-audit.json)

| 切分 | 問題數 | 原始段落與文章家族數 | 原文 token 數 | 不重複的停點數 |
| --- | ---: | ---: | ---: | ---: |
| 訓練 | 260 | 239 | 69,070 | 3,447 |
| 官方 dev 留出 | 22 | 19 | 5,018 | 263 |

兩個切分的文章家族、段落識別碼、正規化段落文字和問題識別碼皆無重疊。所有錨點及至少一個後續停點通過分詞檢查。訓練有 14 題共享錨點，同一錨點下各問題的輸入可相同。正式留出有 0 組嚴格同段交換對照，可估計整體定位表現，無法檢驗同一段落不同問題的身分特異性。 [作者資料稽核](results/iirc-seed-later-audit.json)

每條路徑從錨點末端向後，每 16 個 Qwen token 設一個名目停點，並含段落末候選。不合格前綴不入表。每個保留停點都核對獨立分詞的 token、字元偏移及邊界。每個原始段落在正式擷取只有一次前向，可匯出同段多個錨點所需停點的聯集。凍結 Qwen3.5-2B，保存所需停點的 128 維 FP32 單位向量。額外抽樣前綴檢查用於驗證截短、加右方填補以及變更後文時的因果一致性。 [停點建置](semantic_tracking/iirc_seed_later.py) · [擷取](semantic_tracking/iirc_linked_extract.py)

## 讀出與驗收

三組共享同一停點、離散條件事件模型與 L2 懲罰。位置基線用停點序號、從錨點經過的 token 數及步寬。第二組再加入當前停點表徵，第三組另加入當前與錨點向量逐元素乘積。第二組的表徵也包含已讀語境，因此兩組差異估計顯式錨點向量的增益。

主分析比較完整段落的區間負對數概似及累計 Brier。停點越早讀到完整所選支持文字，事件區間越前。若支持尚未完整出現於觀察範圍，則在最後停點右設限。樣本本身依最終確有稍後支持文字篩選，完整段落資料的 282 題皆觀察到事件。另以錨點後 64 個 token 的最後通過停點為固定截短控制。先核對前綴機率與完整預測的相同停點一致，再報截短條件的觀測及設限分數。這個控制只是條件於稍後存在支持文字的較短觀察窗。

沿用連結證據實驗的固定比較門檻，待留出預測封存後才接上標註。以文章家族進行 10,000 次配對重抽樣，候選相對位置基線的概似損失差值區間上界須 ≤−0.02 nats，累計 Brier 差上界須 ≤+0.01。候選相對當前表徵基線的概似損失差上界須 <0。固定前綴控制僅報描述結果。19 個留出文章家族限制效果估計的精度。 [固定門檻及評分](iirc-linked-evidence-protocol.md#指標與驗收門檻)

## 執行與進度

本機已用指定 Qwen 分詞器建置資料。封包 `model-inputs.tar.gz` 有六個成員，SHA-256 為 `e345442379ee14c5e478921314329ecf9caf9e57dc3aca4a16ce9e5157232d3b`。將封包解壓至 `inputs` 後，使用同一套 L4 擷取器。留出實際標註 `dev/locked-intervals.jsonl` 保留在本機。 [準備、擬合及評分程式](semantic_tracking/iirc_seed_later_assay.py)

```sh
python -m research.semantic_tracking.iirc_linked_extract --inputs inputs --output extraction
python -m research.semantic_tracking.iirc_seed_later_assay fit --inputs inputs --features extraction/features.npz --extraction extraction/extraction.json --output fit
python -m research.semantic_tracking.iirc_seed_later_assay score --inputs inputs --outcomes prepared/dev/locked-intervals.jsonl --forecasts fit/predictions.jsonl --fit-seal fit/fit-manifest.json --output scores.json
```

CPU 合成測試先核對三組擬合、固定截短的右設限評分與預測完整性。實際資料用擷取程式的分詞規則重驗 3,710 個停點，資料封包重建逐位元一致。依先前同類型擷取實測 2,419 token/s，配置前估計前向及讀出約 31 秒。 [本次預檢](results/iirc-seed-later-assay-preflight.json) · [前次擷取量測](results/iirc-anchor-run/extraction.json)

## Colab L4 留出結果

固定 Qwen 在 L4 處理 258 段原文、74,088 個 token 和 3,710 個因果停點，前向與讀出耗時 31.10 秒。對 22 個留出問題，位置基線、正文狀態、正文加顯式錨點的區間 NLL 分別為 2.6161、2.5422、2.6185 nats，越低越好。顯式錨點相對只用正文的差為 +0.0762 nats，配對 95% 區間 [−0.0160, +0.1645]。相對位置的 NLL 差為 +0.0024 nats，95% 區間 [−0.0987, +0.0907]，Brier 的上界也超過非劣門檻。三項預定條件均未通過。 [擷取紀錄](results/iirc-seed-later-l4/extraction.json) · [留出評分](results/iirc-seed-later-l4/scores.json)

在預定的錨點後 64-token 截短對照中，8 題已看完標註證據，14 題右設限。顯式錨點相對正文的 NLL 差為 +0.0750 nats，95% 區間 [+0.0078, +0.1418]。此資料只衡量已給定錨點的事後支持何時完整可見。 [截短評分](results/iirc-seed-later-l4/scores.json)
