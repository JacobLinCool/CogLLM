# 合約的證據融合讀出未通過論文留出

在 Evidence Inference 2.0 的英文論文上，合約擬合的融合讀出三項預定門檻均未通過。整篇證據 AP 為 **0.215**，低於每個匹配詞取相同權重的 BM25 **0.306**。差值為 −0.092，配對 95% 區間 [−0.177, −0.014]。同篇證據配對為 0.737 與 0.840，差值區間包含零。這次留出結果限於預先給定問題後，正文證據位置的排序。 [預定門檻](evidence-inference-transfer-protocol.md) · [完整結果](../results/evidence-inference-transfer-l4/results.json) · [原始研究](https://aclanthology.org/2020.bionlp-1.13/)

## 材料與觀測

從官方測試集按固定順序納入 24 篇完整論文及 89 個不同問題，共 134,158 個 tokens。文章最長 8,001 tokens。在取滿以前篩查 116 篇，37 篇少於兩個合格問題，35 篇因權利聲明條件排除，20 篇超出長度或缺少可用文字。選樣只用原文、來源與查詢，未按證據結果調整。凍結候選的 Punkt 切出 3,395 個候選句末，3,378 個能精確對回因果前綴。 [來源稽核與排除](../results/evidence-inference-transfer-l4/source-audit.json) · [分詞預檢](../results/evidence-inference-transfer-l4/preflight.json)

原始資料有 80 題包含 148 份經核准且逐字匹配的人工證據，另 9 題維持未知。3 份證據有不可對齊的非空白字元。AP 納入 23 篇的 79 題、145 份證據，依序等權平均標註、問題及文章。配對納入 17 篇的 97 組問題，要求兩題的合格證據均可對齊，且句子聯集不重疊。所有排序在接合人工標註前封存。 [封存順位](../results/evidence-inference-transfer-l4/rankings-seal.json) · [可評分母](../results/evidence-inference-transfer-l4/results.json)

## 排序對照

| 讀出 | 證據 AP | 同篇證據配對 |
| --- | ---: | ---: |
| BM25，合約 IDF | 0.202 | 0.639 |
| BM25，單位 IDF | **0.306** | **0.840** |
| 原 Qwen 分數 | 0.073 | 0.510 |
| Qwen 扣除同篇共用分數 | 0.097 | 0.824 |
| 只用 Qwen 擬合 | 0.081 | 0.713 |
| 詞彙加 Qwen 原分數 | 0.201 | 0.626 |
| 詞彙加 Qwen 扣除共用分數 | 0.215 | 0.737 |

融合讀出相對合約 IDF 的 AP 差為 +0.013，95% 區間 [−0.030, +0.061]。相對單位 IDF 的同篇配對差為 −0.102，區間 [−0.217, +0.017]。扣除同篇共用分數的 Qwen 在證據配對得 0.824，整篇 AP 仍只有 0.097。配對只比較人工選出的兩套證據，整篇 AP 也納入其他已對齊句子。這兩個結果顯示，Qwen 在兩套已知證據間有順位訊號，但高分句在整篇中未集中到人工證據。 [兩項端點、差值及區間](../results/evidence-inference-transfer-l4/results.json)

補充查核發現，89 題各自至少有一個同時出現在可評句和題幹、卻沒有合約 IDF 的詞。以題目與相同詞計算為 713／1,629 項。有 767 組題目和候選句只匹配這類詞，原合約 BM25 全部給零分。此查核標出原詞彙讀出在新文體的涵蓋缺口。合約係數比例、詞彙覆蓋及問題集合不同的各自貢獻仍待獨立對照。 [固定詞彙統計](../results/contractnli-rerank-l4/fitted.json) · [獨立重算](../results/evidence-inference-transfer-l4/independent-check.json)

## 重現和適用範圍

使用 `evidence_inference_transfer_prepare` 重建正文與來源標註，使用 `evidence_inference_transfer_forecast preflight` 核對原文座標。L4 以同一 Qwen3.5-2B 快照及 QASPER 頭執行，各篇正文一次因果前向，各外部題幹獨立編碼一次。`evidence_inference_transfer predict` 原樣讀取合約模型，封存排名及評分程式雜湊，再以 `score` 接合標註。兩步重跑的預測及結果逐位元一致。 [資料準備](evidence_inference_transfer_prepare.py) · [前向及封存](evidence_inference_transfer_forecast.py) · [順位及評分入口](evidence_inference_transfer.py)

12 項因果檢查通過。獨立程式逐字核對證據覆蓋，所有可評分母完全一致，點估計最大差約 \(1.11\times10^{-16}\)。L4 前向及讀出為 57.85 秒，遠端階段為 140.72 秒，峰值保留 8.67 GiB。資源配置至核對釋放約 215 秒，帳戶顯示消耗約 0.11 CU。兩次本機評分與一次預測重播均低於 0.10 秒，合併程序最高 RSS 約 90 MiB，未載入 Qwen。 [執行與釋放收據](../results/evidence-inference-transfer-l4/run-verification.json) · [擷取收據](../results/evidence-inference-transfer-l4/extraction.json) · [本機重跑](../results/evidence-inference-transfer-l4/local-verification.json)

融合讀出在合約留出通過，這批科學正文的留出未支持跨文體採用。Gradio 的證據顯示須另有合格讀出及實際介面驗證。作者釋出的標註是在閱讀全文後選取證據，無從檢驗讀者最先在哪個 token 卡住，或沿著文字在哪裡解開同一疑問。 [合約留出](contractnli-rerank-results.md) · [人工觀測範圍](evidence-inference-source.md)
