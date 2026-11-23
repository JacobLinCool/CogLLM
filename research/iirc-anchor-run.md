# IIRC 純正文片語定位留出結果

**Qwen 提高了單一及兩個相異片語的覆蓋，兩項皆通過預定比較。** 給模型未見過的純文字段落，在前五個提案中，單一目標的重疊覆蓋比形式基線高 9.67 個百分點，兩個同停點的目標各自得到不同提案的覆蓋高 6.67 個百分點。兩目標同時覆蓋的絕對值仍是 12.71%。 [封存評分](results/iirc-anchor-run/scores.json) · [評估規格](training-protocol.md#iirc-純正文片語定位)

## 資料與模型

IIRC 的提問者看完一段有連結的英語維基百科文章後，提出問題並選出預期可查到答案的連結。擬合取 4,754 篇官方訓練文章中的 7,195 個可唯一對到字元區段的提問。當中 7,017 個目標落在事先限制為一至八個連續詞段的片語候選。採同一候選及非重疊選擇規則，形式基線用長度、相對位置和大小寫等訊號。Qwen 組加上從一次因果前向取出的片語起訖表徵。僅用文章正文作推論，留出目標、連結和問題在預測封存後接上。 [作者實驗 §2](https://aclanthology.org/2020.emnlp-main.86.pdf#page=3) · [資料稽核](results/iirc-preparation-audit.json) · [訓練及封存](results/iirc-anchor-run/fit-manifest.json)

## 未見文章的覆蓋

| 前五個提案的評估 | 形式基線 | 加入 Qwen | Qwen 減基線，百分點及 95% 區間 |
| --- | ---: | ---: | ---: |
| 單一目標，858 個 | 28.21% | 37.88% | +9.67 [+5.85, +13.56] |
| 兩目標由兩個不同提案覆蓋，944 對 | 6.04% | 12.71% | +6.67 [+3.51, +10.13] |
| 單一目標的精確邊界，858 個 | 21.79% | 31.93% | +10.14 [+6.65, +13.65] |

第一、二項的覆蓋門檻為原文字元交並比 ≥0.5，數值越高越好。以來源文章做 10,000 次配對重抽樣。正式留出有 430 篇文章，單一目標的觀測來自其中 372 篇，兩個目標的配對來自 216 篇。獨立程式從作者封存檔重建目標和配對，另重算平均、差值及區間，和正式評分的最大數值差為 5.55×10⁻¹⁷。主指標適用於段落末已提出的問題所選目標。首次感到疑問的時間與後續解答需各自的觀測資料。 [分數及區間](results/iirc-anchor-run/scores.json) · [獨立重算](results/iirc-anchor-run/independent-verification.json)

## 執行與封存

Colab L4、Transformers 5.17.0 和 FP32 Qwen 擷取 5,184 篇段落的 1,464,005 個 token，無段末排除，12 項前綴檢查通過。模型前向及讀出累計 605.11 秒，CUDA 峰值保留約 8.50 GiB。整段擷取指令執行 667.88 秒，兩組擬合指令 20.52 秒。L4 已確認釋放。主跑分配確認至送出停止要求為 739.67 秒，期間另有兩個帳戶活動指派，總餘額由 758.18 降至 756.88 CU。先前預檢的清單核對有型別錯誤，該次資源亦已釋放。 [擷取量測](results/iirc-anchor-run/extraction.json) · [主跑紀錄](results/iirc-anchor-run/manifest.json) · [用量](results/iirc-anchor-run/usage.jsonl) · [首輪預檢](results/iirc-anchor-run/initial-run-termination.json)

主跑的初始本機清單比實際少兩個 token。使用與 L4 相同的 AutoTokenizer 及固定模型版本重播，差異縮至一篇包含天城文組合字元的段落，重新確認 1,464,005 個 token 和全部段末。以獨立重播的清單重新驗證原封存成果後才評分。預測、權重、輸入和評分規則的雜湊均未改動。評分重跑逐位元一致。 [分詞重播](results/iirc-anchor-run/token-inventory-replay.json) · [封存重驗](results/iirc-anchor-run/revalidation.json) · [讀出權重](results/iirc-anchor-run/heads.npz) · [留出預測](results/iirc-anchor-run/predictions.jsonl)

重現時先用 `iirc_prepare` 建置作者原始資料，再依序用 `iirc_extract`、`iirc_anchor_fit` 和本機的 `iirc_anchor_score`。訓練資料以原文 `paragraph_id` 及半開 `start`、`end` 區段提供一筆已觀測提問的標註。留出推論僅提供 `id`、`text`、`stops`。 [資料建置](semantic_tracking/iirc_prepare.py) · [模型擷取](semantic_tracking/iirc_extract.py) · [讀出擬合](semantic_tracking/iirc_anchor_fit.py) · [留出評分](semantic_tracking/iirc_anchor_score.py)
