# 自適應聯合訓練的實驗結果

在未設保留約束時，按來源自適應加權在兩個種子都降低了等權開發 NLL，但既有逐 token 曲線退步。同時約束共享殘差及保留舊曲線係數時，固定和自適應兩版都保留了早解與晚解的定位表現。相同約束下，自適應版在兩個種子的等權開發 NLL 略高於固定版。正式人類效應方向和幅度仍有欠缺。 [原比較](../results/adaptive-joint-l4/training-comparison.json) · [帶約束配對](../results/adaptive-joint-retention-l4/retention-comparison.json)

## 資料與自適應方法

參考新版 Qwen 逐 token 讀出，將原本 24 組來源的 62 個觀測區塊接回共享狀態，另外納入 Frick 的 token 聚合讀出、ExpMRC 和 IIRC，共 26 個來源、65 個 loss 區塊。舊資料使用 135,598 筆訓練觀測和 33,208 筆開發觀測。221 筆舊版 Frick 觀測移至新版的開發側，219 筆與新版 Frick 測試重合的舊觀測保留在本機評分側。每個來源保留資料原本的尺度、有效分母與因果停止位置。 [量測規格](../union_training/observation-results.md) · [本輪封包收據](../results/adaptive-joint-l4/input-receipt.json)

一份原文可有多種讀出。同來源的多個區塊共享來源權重，Frick 的兩個首次回報投影因而沒有額外增加該來源的總權重。開發選取按具開發量測的來源等權，IIRC 和 Qiao 只提供訓練訊號。自適應法用各量測端點自己的暖身損失尺度與先前量到的共享梯度強度更新權重。每個最優化步的總權重與固定組相同。單次反向同時取得參數梯度與自適應所需的共享梯度。兩個種子中，ExpMRC 的實際權重約為固定組的 0.55 倍，IIRC 約為 0.50–0.52 倍，SCARECROW 約為 1.45–1.49 倍。 [研究規格](protocol.md) · [權重統計與來源分數](../results/adaptive-joint-l4/training-comparison.json)

共同封存的 164,220 個最後一層狀態從 18 個原有分片重播。確認快照一致後，將 2,048 維狀態投影為 48 維，無需重新執行 27,167 次原文前向。逐 token 端點的 351 篇在本輪 L4 各前向一次。新讀出使用共享 48×48 殘差，匯出時併入既有讀出投影，每個權重的分數均獨立重算。整個配對工作階段從配置到釋放為 441 秒。 [前向收據](../results/adaptive-joint-l4/causality.json) · [L4 釋放](../results/adaptive-joint-l4/run-verification.json)

## 訓練與留出結果

| 種子 | 固定最佳開發 NLL ↓ | 適應最佳開發 NLL ↓ | 固定與適應到相同 NLL 的輪數 | 開發效應方向一致，固定 → 適應 |
| --- | ---: | ---: | --- | --- |
| 1729 | 1.315 | 1.294 | 8 → 8 | 17／34 → 19／34 |
| 2718 | 1.314 | 1.297 | 8 → 8 | 23／34 → 23／34 |

兩組固定臂於第 15 輪開發停訓，各自適應臂於第 18、19 輪停訓。在對齊的第 15 輪，自適應用時較固定少 0.44 秒和多 0.004 秒。兩組每輪均做 160 次更新及 640 次反向。梯度裁剪比例介於 0.66% 與 0.83%，兩種策略相近。自適應改善約 0.019 nats 的配對平均最佳開發 NLL。34 項已登記的可重播開發效應中，幅度誤差的來源重抽樣 95% 區間在兩個種子都跨零。 [訓練核對](../results/adaptive-joint-l4/training-comparison.json) · [種子 1729 效應](../results/adaptive-joint-l4/evaluation/seed-1729/development-comparison.json) · [種子 2718 效應](../results/adaptive-joint-l4/evaluation/seed-2718/development-comparison.json)

正式驗收使用整組未進本輪訓練的歷史資料。SAP 在消歧詞的方向命中為 3／3 → 2／3，以及 0／3 → 0／3。Anglin 兩個種子均從 3／6 維持在 3／6，六項的幅度皆未進入預定容差。繁中 MECO 的 1,440 次理解作答 NLL 從 0.663 → 0.646 和 0.644 → 0.614，兩次配對區間均低於零。兩個自適應候選高於 0.482 的來源基線。這三個家族在先前模型已有評分，本輪沿用其材料與反應。種子反映訓練變異。 [正式預測封存與無重疊](../results/adaptive-joint-l4/evaluation/formal-prediction-lock.json) · [逐項驗收](../results/adaptive-joint-l4/evaluation/summary.json)

未加保留約束時，固定和自適應兩臂都將初始候選的早解／晚解雙命中從 6／12 降至 0／12。ExpMRC 留出區間 NLL 從初始 2.772 升至 3.917–4.541（自適應）或 4.121–4.303（固定）。雙疑問的完整雙向診斷在所有臂仍為 0／8。這兩個說明文診斷是自編控制，真人端點以實際觀測為準。 [逐 token 結果](../results/adaptive-joint-l4/token-evaluation.json)

## 保留約束與相同條件下的固定對照

原來逐 token 讀出的係數固定，所有來源仍可更新共享的 48×48 殘差。每次更新將殘差的 Frobenius 範數限制在 0.10，並依兩個舊 token 端點的開發損失選取候選。固定和自適應兩組的四個候選均通過開發保留門檻。所有暫定候選都落在第 28 輪預算邊界，因此尚未達到開發停訓。 [保留規格](retention.md) · [同約束控制](retention-control.md) · [逐輪訓練配對](../results/adaptive-joint-retention-l4/retention-comparison.json)

| 保留約束下的種子 | 固定開發 NLL ↓ | 自適應開發 NLL ↓ | 開發效應方向一致，固定 → 適應 | 兩版的早解／晚解雙命中 |
| --- | ---: | ---: | --- | --- |
| 1729 | 1.1944 | 1.1956 | 14／34 → 20／34 | 6／12 |
| 2718 | 1.1974 | 1.1992 | 19／34 → 19／34 | 6／12 |

兩個種子的固定保留比固定無約束的等權開發 NLL 低。帶約束的兩種策略用相同的 4,480 次更新和 17,920 次反向，自適應的配對用時多 1.18 秒和 0.25 秒。四臂的梯度裁剪比例為 0.63%–0.76%。ExpMRC 區間 NLL 為 2.761–2.765，接近初始候選的 2.772，雙疑問仍為 0／8。原本曲線的保留可在固定權重下重現。 [守門固定逐 token 結果](../results/adaptive-joint-fixed-retention-l4/token-evaluation.json) · [守門自適應逐 token 結果](../results/adaptive-joint-retention-l4/token-evaluation.json)

同約束下，SAP 的消歧詞命中在兩個種子分別為 1／3 和 0／3，兩種策略相同。Anglin 的方向均為 3／6，種子 1729 兩種方法均沒有幅度命中，種子 2718 兩種方法均有 1／6。繁中 MECO 的 NLL 在固定與自適應版分別為 0.569 → 0.591、0.519 → 0.520。種子 1729 的同約束配對區間顯示自適應版誤差較高，種子 2718 區間跨零。這項續測在看到第一次的逐 token 退步後設計，並重用同一批正式評分材料。 [同約束的人類效應](../results/adaptive-joint-retention-l4/evaluation/summary.json) · [兩次封存與樣本隔離](../results/adaptive-joint-retention-l4/evaluation/formal-prediction-lock.json)

## 重現及產物

用 `research.adaptive_joint.cache` 重建封包，`research.adaptive_joint.colab` 的 `prepare` 與 `run` 提交訓練。原始比較使用預設參數，保留版分別在兩個子命令加 `--guarded` 或 `--guarded-fixed`。用 `research.adaptive_joint.verify` 核對輸出雜湊和 L4 釋放，`research.adaptive_joint.evaluate` 在本機重播真人效應，`research.adaptive_joint.token_evidence` 重播 token 評分。三次訓練工作階段均確認釋放。 [原兩臂執行](../results/adaptive-joint-l4/run-verification.json) · [保留自適應執行](../results/adaptive-joint-retention-l4/run-verification.json) · [保留固定執行](../results/adaptive-joint-fixed-retention-l4/run-verification.json)

按開發 NLL 選出的聯合讀出為帶保留約束的固定權重種子 1729。該權重用既有 portable runner 產生逐 token 曲線。訓練時的共享殘差已合併至投影，權重的三項曲線分數通過 L4 上的數值重播。 [候選曲線權重](../results/adaptive-joint-fixed-retention-l4/guarded-fixed-1729/curve.pt) · [選取及數值驗證](../results/adaptive-joint-fixed-retention-l4/guarded-fixed-1729/fitting.json)
