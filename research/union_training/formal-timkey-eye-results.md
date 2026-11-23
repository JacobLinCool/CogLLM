# Timkey 逐詞眼動結果

OneStop 校準的 CogLLM 眼動讀出在 Timkey 的四項預定主比較中，三項效果方向與真人相同，但幅度皆偏小。首次離開時回視的效果方向相反。四項逐次反應的負對數似然都高於只用詞長及位置的對照，四個模型減對照的 95% 材料重抽樣區間都在零以上。 [正式結果](../results/union-observations-formal-l4/timkey/scores.json) · [固定規格](formal-timkey-eye.md)

| 主端點 | 詞位 | 真人歧義效應［95% 區間］ | CogLLM 效應 | CogLLM／詞長與位置 NLL |
| --- | --- | ---: | ---: | ---: |
| 首次離開時回視 | 消歧詞下一詞 | +19.10［+16.01, +22.05］百分點 | −0.24 百分點 | 0.6752／0.6728 |
| 向前閱讀時間 | 消歧詞 | +13.75［+2.62, +23.42］ms | +0.30 ms | 0.8505／0.8396 |
| 回視前注視時間 | 消歧詞 | +48.31［+26.33, +71.30］ms | +0.91 ms | 1.1152／1.1100 |
| 回視路徑時間 | 消歧詞下一詞 | +673.75［+563.80, +781.33］ms | +2.93 ms | 1.3587／1.3448 |

每個端點都涵蓋 72 組配對，差值為歧義版減控制版。以材料編號重抽樣，保留同一編號在各構式中的觀測，區間條件於已觀察的讀者群。每項真人效果的區間皆高於零，每項模型減真人的區間皆低於零。負對數似然依各端點的合格觀測計算，數值較低較佳。 [逐端點及分構式結果](../results/union-observations-formal-l4/timkey/scores.json)

L4 對 339 份 OneStop 顯示文本各做一次 forward，取得 36,983 個已驗證的詞末前綴。眼動頭只以 OneStop 初讀的合格反應擬合，四個端點分別有 472,694、357,562、115,132 與 115,132 筆。最終擬合的最大梯度小於 3 × 10⁻⁸。逐批留出時，CogLLM 相對詞長與位置對照改善三項的平均負對數似然。這項留出屬於眼動量測頭。先前的共享狀態訓練已使用 OneStop 家族的閱讀理解資料。 [L4 擷取驗證](../results/union-observations-formal-l4/timkey/extraction-verification.json) · [OneStop 眼動讀出](../results/union-observations-formal-l4/timkey/forecast/fit.json)

Timkey 的 288 個詞位與作者獨立發布的字詞表逐一確認後，沿用 Huang SAP 共用材料已擷取的狀態。Timkey 的反應只在預測封存後用於評分。結果顯示，凍結的共享狀態加上 OneStop 校準，仍不足以重現這組未見材料的回視機率或延長幅度。這個驗證的範圍是經典句法歧義構式中的眼動反應。 [獨立詞位稽核](../results/union-observations-formal-l4/timkey/timkey-roi-alignment-independent.json) · [預測封存](../results/union-observations-formal-l4/timkey/forecast/seal.json) · [Timkey 等原研究](https://www.pnas.org/doi/10.1073/pnas.2532230123)
