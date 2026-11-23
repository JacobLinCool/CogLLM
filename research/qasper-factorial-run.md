# QASPER 逐 token 交互與排序監督對照

**新論文上，原疑問比交換疑問更能定位標註證據，但正式候選未達同停點排序和校準標準。** 候選使用逐 token 交互讀出及同篇排序監督。在 27 篇未評過的 QASPER 論文中，原疑問的區間負對數概似比同篇交換疑問低 0.0830 nats，配對 95% 區間為 [−0.1334, −0.0299]。同停點的勝出有 29／49 對，對位置及只看疑問的兩個基線，差值區間皆跨零。這一輪保留段落讀出作參照。 [原始分數](results/qasper-factorial-run/scores.json) · [固定驗收](training-protocol.md#新-qasper-論文的固定評分)

## 設計

用原 60 篇訓練論文、151 個疑問和 121 對同篇排序，固定 Qwen3.5-2B、段末及段平均讀出和 128 維投影。四組各訓練 518 個係數。逐 token 版本在每段內，用當時可見 token 對問題計算交互分數，以 log-mean-exp 取代平均表徵的局部交互。稍早段落的訊號進入因果段末狀態和累積到達機率。加排序的兩組各用 0.5 倍同篇同停點排序損失，所有組都用證據區間似然。 [讀出、配對及訓練程式](semantic_tracking/qasper_factorial_fit.py) · [擬合紀錄](results/qasper-factorial-run/fit-manifest.json)

正式留出來自官方 dev 的 27 篇論文、65 個疑問與 49 個嚴格同篇配對。原訓練、原測試和新留出在論文、題目識別碼及正規化標題皆無重疊。留出證據由獨立標註者看完整篇後選出。讀出收到既有疑問，評分停點是證據所在或比較用的段末。此測量涵蓋指定問題的文字證據到達，真人首次感到困惑和察覺疑問解除仍需配對的閱讀中量測。 [資料與對齊](results/qasper-factorial-run/audit.json) · [QASPER §2.2](https://aclanthology.org/2021.naacl-main.365.pdf#page=3)

## 留出結果

| 讀出 | 同停點排序，49 對 | 配對證據區間 NLL，nats／方向 ↓ | 累積機率 Brier ↓ |
| --- | ---: | ---: | ---: |
| 只看位置 | 50.0%（49 對平手） | 3.1235 | 0.17310 |
| 只看疑問 | 44.9%（22 對） | 3.6363 | 0.20492 |
| 段落＋區間 | 61.2%（30 對） | 2.9322 | 0.16872 |
| 段落＋區間及排序 | 59.2%（29 對） | 2.8977 | 0.16513 |
| 逐 token＋區間 | 59.2%（29 對） | 2.9500 | 0.16852 |
| 逐 token＋區間及排序（正式候選） | 59.2%（29 對） | 2.9141 | 0.16464 |

同停點排序將較早證據的疑問給較高到達機率計為勝，平手計半。NLL 越接近零代表分配給標註區間的機率越高。Brier 是累積到達機率與二元標註的均方差，範圍 0–1，零為最佳。全部配對區間以論文為單位做 10,000 次重抽樣。 [逐項估計](results/qasper-factorial-run/scores.json)

- **排序損失改善逐 token 讀出的區間機率。** 在逐 token 組，加入排序使配對 NLL 差為 −0.0359 nats，95% 區間 [−0.0640, −0.0045]，兩組的同停點命中均為 29／49。已用排序時，逐 token 對段落讀出的 NLL 差為 +0.0164 nats [−0.0094, +0.0336]。留出結果對逐 token 表徵的額外收益仍不確定。 [四組配對比較](results/qasper-factorial-run/scores.json)
- **正式候選通過五項中的兩項。** 原疑問優於交換疑問，Brier 相對段落＋區間基線的差為 −0.00408 [−0.00803, −0.00062]，落在容許增加量內。同停點排序相對位置和只看疑問的差分別為 +9.18 個百分點 [−8.93, +26.19] 和 +14.29 個百分點 [−4.44, +35.19]。平均機率偏差為 +5.70 個百分點 [−1.70, +12.86]，超出預定的 ±5 個百分點區間標準。 [正式驗收結果](results/qasper-factorial-run/scores.json)

## 執行與重現

Colab NVIDIA L4 上用 Transformers 5.17.0、PyTorch 2.14.1 和 FP32 骨幹，擷取 235,523 個有效 token，其中 87 篇文章的 233,359 個 token 保存投影狀態。全部 2,280 個停點對齊。12 項投影因果對照通過，歷史前綴的最大相對 RMS 差為 1.44×10⁻⁵。擷取的模型前向及讀出共 102.78 秒，CUDA 峰值保留 9.39 GiB，六組讀出在擬合程式內耗時 3.54 秒。L4 配置確認至發出釋放要求約 206.40 秒，已確認釋放。全帳戶活動還包括其他執行，期間餘額從 759.20 降至 758.82 CU。 [擷取](results/qasper-factorial-run/extraction.json) · [因果檢查](results/qasper-factorial-run/causality.json) · [執行紀錄](results/qasper-factorial-run/manifest.json) · [用量快照](results/qasper-factorial-run/usage.jsonl)

依序使用 `qasper_prepare --holdout dev`、`qasper_extract --dense-tokens`、`qasper_factorial_fit` 和 `qasper_factorial_score`。留出標註留在本機，GPU 收到的留出配對只含識別碼。分數重跑逐位元一致，另以直接連乘存續機率的算式重算 36 個指標。 [封存權重](results/qasper-factorial-run/heads.npz) · [封存預測](results/qasper-factorial-run/predictions.jsonl) · [獨立重算](results/qasper-factorial-run/independent-verification.json)
