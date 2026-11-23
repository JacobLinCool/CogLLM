# 繁體中文 MECO 篇後理解驗證

Qwen3.5-2B 的第 31 epoch 聯合讀出在繁中 MECO 的篇後理解正確率上，零樣本轉移表現低於固定基線。每次作答平均負對數概似高出 **0.140 nats**，按讀者與文章交叉重抽樣的 95% 區間為 **[0.070, 0.206]**。這組 12 篇文章支持轉移變差。 [評分原檔](../results/union-observations-formal-l4/meco/scores.json) · [評分規格](formal-meco-plan.md)

| 指標 | CogLLM | OneStop 固定基線 |
| --- | ---: | ---: |
| 平均 Bernoulli 負對數概似，nats／作答，低者較好 | 0.622 | 0.482 |
| Brier score，範圍 0–1，低者較好 | 0.215 | 0.152 |
| 平均預測答對率 | 56.12% | 80.05% |
| 平均預測答對率減真人答對率，百分點 | −25.20 | −1.27 |

資料包含 60 位香港讀者、12 篇繁中說明文及 1,440 次橫排篇後是非作答，真人答對 1,171 次（81.32%）。模型僅讀正文，每篇在 Colab L4 執行一次因果 forward，並用英文 OneStop 訓練的初讀正確率讀出在篇末產生單一機率。固定基線取 OneStop 訓練折的 1,208／1,509 次答對。這次檢驗的範圍是該讀出跨語言、讀者群與題型的篇後理解轉移。第 6 篇的 120 次作答中有 115 次答對（95.83%），模型給該篇 56.23%。 [研究方法及資料](https://doi.org/10.1038/s41597-026-06989-8) · [逐篇結果](../results/union-observations-formal-l4/meco/scores.json)

在讀取逐次反應前已鎖定 12 個預測。初次計分的資料鍵檢查停止執行，因為 RDA 的 `Session_Name_` 編碼讀者與呈現方向，篇章應按 `number` 與實際呈現檔名對位。修正後在原始設計欄位上用合成反應驗證對位。兩次中止均未產生分數，修正後的預測檔和原鎖定檔逐位元組相同。對 27,167 篇開發及訓練材料的查核，完整正文與連續 24 字元比對皆無重疊。 [原始鎖定](../results/union-observations-formal-l4/meco/forecast/forecast-seal.json) · [對位更正紀錄](../results/union-observations-formal-l4/meco/adapter-verification.json) · [文字重疊查核](../results/union-observations-formal-l4/meco/training-text-overlap.json)
