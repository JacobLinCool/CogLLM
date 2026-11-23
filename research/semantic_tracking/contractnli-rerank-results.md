# 詞彙與 Qwen 融合的證據排序通過留出驗收

在 23 份未用於擬合的 EDGAR 文件上，詞彙與 Qwen 融合的讀出通過三項預定門檻。整篇證據 AP 為 **0.502**，高於 BM25 的 0.440 和只用 Qwen 擬合的 0.245。文章相同而命題改變時，兩組人工證據的配對分數由 BM25 的 0.701 升到 **0.809**。這批資料量測指定命題所對應的文字證據。 [固定規格](contractnli-rerank-protocol.md) · [封存留出結果](../results/contractnli-rerank-l4/results.json)

## 擬合和留出

用前一輪完成評分的 47 篇 EDGAR 文件訓練讀出。每組只估兩個係數，使用同樣的訓練尺度、等權文件家族和 ridge。預先指定的驗收組將訓練候選庫固定的 BM25 詞彙分數，加上逐 token Qwen 分數扣除同篇、同停點 17 個預先提供命題的共用分數。兩個係數在訓練標準化量尺上為 1.724 和 1.120。固定詞彙統計與凍結權重，從每篇一個 Qwen 因果前向就能平行取得全部候選的模型特徵。 [訓練材料](contractnli-evidence-results.md) · [係數與訓練詞彙統計](../results/contractnli-rerank-l4/fitted.json)

新留出為官方 dev 分割的 23 篇文件和 391 個文章×命題組合。47 篇訓練與 23 篇留出的識別碼、原始及正規化正文、來源網址均不重疊。跨組字元及詞彙 5-shingle Jaccard 最大值分別約 0.796 和 0.704，低於固定的 0.80 界線，仍可能共用書寫模板。去掉 54 個純空白候選及 1 個無法對齊的候選，留下 1,684 個可評位置。一筆原始標註把空格列入完整證據集合，因此 235 個有證據的命題中有 234 個納入 AP，留出的原始人工欄位完整保存。 [跨組材料稽核](../results/contractnli-rerank-l4/source-audit.json) · [tokenizer 預檢](../results/contractnli-rerank-l4/preflight.json)

## 留出結果

AP 衡量人工證據是否集中在排行前段，配對分數衡量同篇中較高分的位置是否隨命題改變。兩項皆為 0–1，越高越好。配對分數以 20 篇文件上的 308 組支持／反駁命題、616 個比較方向計算。表內數值先在文件內平均，再以文件家族等權。 [逐項統計](../results/contractnli-rerank-l4/results.json)

| 讀出 | 全文證據 AP | 同篇命題配對 |
| --- | ---: | ---: |
| BM25 | 0.440 | 0.701 |
| Qwen 原分數 | 0.168 | 0.530 |
| Qwen 扣除共用分數 | 0.227 | 0.863 |
| 只用 Qwen 擬合 | 0.245 | 0.844 |
| 詞彙加 Qwen 原分數 | 0.435 | 0.700 |
| **詞彙加 Qwen 扣除共用分數** | **0.502** | **0.809** |

融合 AP 相對 BM25 增加 **0.062**，配對 95% 區間 [0.034, 0.093]，相對只用 Qwen 擬合增加 **0.256** [0.212, 0.303]。同篇命題配對相對 BM25 增加 **0.108** [0.068, 0.150]。與等容量的詞彙加 Qwen 原分數相比，融合 AP 差為 +0.066 [0.040, 0.097]。只扣共用分數的配對分數較高，與融合的配對差區間包含零。這組結果支持用兩種訊號做整篇候選排序，也保留了不同端點對讀出組合的要求。 [完整比較與區間](../results/contractnli-rerank-l4/results.json)

這個讀出驗收的是有 17 個固定命題與原始候選位置的英文合約。它輸出排序分數，沒有在這次實驗估計絕對理解機率。跨領域、開放命題、正文自發疑問及讀到哪裡才解開仍需要各自的留出證據。 [規格及適用範圍](contractnli-rerank-protocol.md)

## 重現與資源

跑 `research.semantic_tracking.contractnli_rerank_prepare` 準備留出材料，以 `research.semantic_tracking.contractnli_rerank_forecast` 認證停點及擷取凍結分數。`research.semantic_tracking.contractnli_rerank` 的 `fit`、`predict`、`score` 依序封存權重、預測與結果。三步各重跑一次均逐位元相同。另一段程式從原始詞彙計數、Qwen 分數與封存係數獨立重建 391 組排名，最大數值差約 \(1.38\times10^{-14}\)。信賴區間最大差為 \(1.11\times10^{-16}\)。八篇快取資料的變更後文對照中，6,120 個共有前綴特徵向量均不受變更影響。 [程式](contractnli_rerank.py) · [驗證收據](../results/contractnli-rerank-l4/run-verification.json)

Colab L4 處理 53,121 個正文與命題 token、1,724 個認證停點，12 項因果檢查通過。擷取階段前向及讀出為 24.10 秒，遠端階段為 81.38 秒，GPU 峰值保留約 9.07 GiB。L4 已確認釋放。本機擬合、預測、評分分別約 0.41、0.30、0.26 秒，最高 RSS 約 119 MiB。 [GPU 擷取](../results/contractnli-rerank-l4/extraction.json) · [資源快照](../results/contractnli-rerank-l4/usage.jsonl)
