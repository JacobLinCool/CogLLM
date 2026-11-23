# Currie 篇末一致性判斷的跨研究留出結果

Frick 的配對讀出未通過四項跨研究門檻。Currie 等人的六年級讀者在矛盾敘事文中回報「說不通」的機率，比矛盾說明文高 **24.61 個百分點**。模型的平均差是 **−0.22 個百分點**，且兩種文體的篇末回報機率都估得偏低。結果涵蓋 24 個新題材、48 份英文文本與 4,248 筆既有讀後判斷。 [事先固定的規格](larrc-consistency-protocol.md) · [完整評分](../results/larrc-consistency-transfer-l4/results.json)

## 文體差與分條件回報

每篇正文執行一次同一凍結 Qwen3.5-2B 的因果前向。首次事件讀出的 20 組權重及其方向全部沿用 Frick。在各句估計 hazard，逐篇算出篇末曾回報的機率，再平均 20 組。Currie 的真人作答只用於封存後評分。以下比例在每個版本先平均讀者，再等權平均題材。效應是敘事文減說明文，區間按文體內題材及共同讀者交叉重抽樣。 [封存預測](../results/larrc-consistency-transfer-l4/forecast-seal.json) · [資料對齊](larrc-consistency-source.md)

| 觀測或讀出 | 敘事文，近／遠 | 說明文，近／遠 | 敘事減說明，百分點及 95% 區間 |
| --- | ---: | ---: | ---: |
| 真人 | 65.17%／64.89% | 40.05%／40.79% | +24.61 [13.07, 36.28] |
| 全部觀測等權 | 17.15%／17.14% | 18.35%／18.37% | −1.22 [−2.13, −0.32] |
| 目標句加權 | 34.85%／34.92% | 40.33%／40.51% | −5.53 [−9.38, −1.67] |
| 加入配對效應 | 8.61%／9.40% | 9.04%／9.41% | −0.22 [−3.46, 3.36] |

配對讀出的模型減真人差為 −24.82 個百分點，95% 區間 [−37.30, −12.61]，超出 ±10 個百分點的門檻。20 組只有句序差的材料中，真人的遠減近差為 +0.20 個百分點 [−4.86, 5.43]，配對讀出為 +0.53 [−0.27, 1.47]。這是次要的篇末距離對照。 [逐條件分數及句序子集](../results/larrc-consistency-transfer-l4/results.json)

## 逐試次機率

負對數概似（NLL）以 nats／試次計，最低為零，越低代表給實際答案的機率越高。配對讀出的 NLL 是 1.3812，等權讀出為 1.0112，目標句加權為 0.7638。Frick 整篇曾回報率構成的固定對照為 0.7154。配對相對等權的差為 +0.3699 [0.2540, 0.4890]，相對常數差為 +0.6658 [0.4847, 0.8541]，兩者均超過 +0.02 的非劣上界。Frick 回溯分析中加大的條件效應，未帶來這項英文留出的群體差異或機率校準。 [Frick 效應](frick-pair-results.md) · [兩項配對 NLL](../results/larrc-consistency-transfer-l4/results.json)

本次同時改變語言、文本、讀者與提問時點。Frick 逐句按鍵在每篇平均約有 14.6 個可評停點，Currie 模型每篇使用七個 hazard 停點。較短的篇章由相同 hazard 累積出的事件機率會較低。實驗只能判定這套直接移轉失敗，沒有分別操弄篇長、讀者年齡或按鍵時點。這 48 篇 Currie 文本也用於較早的同語料六折實驗。本次凍結的 Frick 讀出及校準使用的是另一批德語故事。Currie 的原始答案只量測篇末。 [兩個觀測設計](larrc-consistency-protocol.md) · [早期同語料六折](../stage-5-semantic-run.md#留出結果) · [原 Frick 分母](../results/frick-pair/results.json)

## 重現與資源

準備檔與原始公開檔案的雜湊、384 個句末分詞驗證、原模型權重、評分程式及釋放確認均已封存。推論入口為 `python3 -m research.semantic_tracking.larrc_consistency_score forecast`，讀取刺激封包和封存 L4 狀態。之後以同模組的 `score` 接合本機人類答案。預測與評分各重跑一次，逐位元一致。另獨立重算 1,008 個逐句累積機率與兩個真人端點，最大數值差約 \(2.22\times10^{-16}\)。 [來源雜湊](../results/larrc-consistency-transfer-l4/source-audit.json) · [分詞驗證](../results/larrc-consistency-transfer-l4/extractor-preflight.json) · [程式](larrc_consistency_score.py) · [獨立重算](../results/larrc-consistency-transfer-l4/independent-check.json)

L4 用 2.87 秒處理 6,258 個正文 token 及讀出，擷取階段為 43.15 秒，峰值 GPU 保留 8.06 GiB。384 個 FP32 句末狀態與 12 個因果檢查均已驗證。執行個體從分配到釋放確認約 123 秒。帳戶執行個體數由 0 到 1 再回到 0，CU 顯示值維持 734.91，這個讀數的精度無法分離單次的實際扣用。本機只有快取狀態的預測和 CPU 評分，兩步合計約 0.48 秒，峰值約 78 MiB。 [擷取收據](../results/larrc-consistency-transfer-l4/projection/seal.json) · [執行與釋放](../results/larrc-consistency-transfer-l4/run-verification.json) · [用量快照](../results/larrc-consistency-transfer-l4/usage.jsonl)
