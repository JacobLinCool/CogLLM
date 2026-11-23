# ContractNLI 的獨立證據定位結果

**凍結的逐 token 讀出通過四項預定判定中的一項。** 在 47 份留出文件、487 個有人工證據的命題上，逐 token 交互讀出的證據 AP 為 0.114，段落交互讀出為 0.105，詞彙比對 BM25 為 0.409。逐 token 的 AP 比段落讀出高 0.009，95% 區間 [0.004, 0.015]。同篇命題換成另一組證據時，兩種 Qwen 讀出的表現接近。 [固定驗收規格](contractnli-evidence-protocol.md) · [封存結果](../results/contractnli-evidence-l4/results.json)

## 指定命題的證據順位

47 篇全文一次因果前向，17 個公開命題各擷取一次並共用於所有文章。固定 Qwen3.5-2B 骨幹及 QASPER 頭，將原文候選末尾的空白修剪後保留字元對回。4,318 個原候選中，102 個只有空白，9 個無法精確對齊 tokenizer，4,207 個可評。487 個有證據的命題與 718 組證據互不重疊的支持／反駁配對都保留。 [原材料與統計單位](contractnli-evidence-source.md) · [分詞檢查](../results/contractnli-evidence-l4/preflight.json)

| 讀出 | 證據 AP，95% 區間 | 同篇配對分數，95% 區間 |
| --- | ---: | ---: |
| 位置 | 0.096 [0.077, 0.118] | 0.500 [0.500, 0.500] |
| 僅命題 | 0.037 [0.031, 0.044] | 0.500 [0.500, 0.500] |
| 段落交互 | 0.105 [0.087, 0.124] | 0.529 [0.518, 0.541] |
| 逐 token 交互 | 0.114 [0.094, 0.136] | 0.527 [0.517, 0.540] |
| BM25 | 0.409 [0.375, 0.446] | 0.737 [0.701, 0.772] |

AP 越高表示人工證據在排序中越集中於前面，最高為 1。同篇配對分數也是 0–1，比較用同一命題評估兩套證據時，自身證據能否排在另一命題的證據之前。兩個對調方向各占一半，共 1,436 個方向。逐 token 相對段落讀出的差為 −0.0013 [−0.0098, 0.0067]，相對 BM25 為 −0.2094 [−0.2476, −0.1706]。此比較用人工標註的證據集合界定哪些位置應隨命題改變，未評分模型對命題本身是否成立的判斷。 [各分項及配對差值](../results/contractnli-evidence-l4/results.json)

## 去掉文章內共用的位置分數

主要驗收完成後，將每篇文章每個停點的 17 個命題 logit 平均視為共用位置分數。從每個命題的原 logit 減去該分數，再經 logistic 轉換，用相同候選及重抽樣評分。這個診斷沿用全部固定命題，未用人工答案計算共用分數。 [診斷程式](contractnli_centered_diagnostic.py) · [診斷結果](../results/contractnli-evidence-l4/centered-diagnostic.json)

去除共用分數後，配對分數由 0.527 升至 **0.854**，差 +0.327 [0.291, 0.360]，也高於 BM25 的 0.737，差 +0.118 [0.073, 0.164]。AP 由 0.114 升至 0.216，仍低於 BM25，差 −0.193 [−0.223, −0.164]。結果顯示在兩套人工已標記的證據之間，表徵有能區分命題的訊號。從整篇未標記候選中找出證據仍更難。這是已評資料上的補充對照，且使用同篇的 17 個固定命題。單一新命題沒有相同的共用分數可以直接扣除。 [差值區間](../results/contractnli-evidence-l4/centered-diagnostic.json)

## 讀到完整標記證據的機率

在 799 個命題中，487 個有人工選定的證據，312 個標為未提及。用最後一個已標註區段的終點計算事件 NLL，用篇末剩餘機率評未提及。NLL 單位為 nats／命題，越低越好，理想值為零。逐 token 原讀出為 **8.530**，原 ExpMRC 移轉校準為 **29.581**，QASPER 混合標註擬合的時間校準為 **3.272**。原 ExpMRC 校準的篇末平均剩餘機率為 0.002，對未提及的 NLL 為 57.831。QASPER 時間校準把這兩項改為 0.451 與 1.173。加入固定剩餘機率項後整體 NLL 為 3.279。這些機率對照只用封存權重，且為補充結果。 [條件機率與 Brier 誤差](../results/contractnli-evidence-l4/results.json)

人工標註只指定全文裡應採的證據。例如兩段各自可提供相同證據時，停點定義仍取標記的最後一段。這個實驗可核對文字何處出現；真人何時察覺矛盾、形成疑問或認為已經解開，仍需要讀者在相應位置的回報。 [觀測定義與留出](contractnli-evidence-protocol.md) · [原始標註規則](https://aclanthology.org/2021.findings-emnlp.164.pdf#page=2)

## 重現與資源

資料準備執行 `research.semantic_tracking.contractnli_prepare`，分詞驗證及遠端推論用 `research.semantic_tracking.contractnli_forecast`，本機評分用 `research.semantic_tracking.contractnli_report`。同一份預測評分兩次逐位元一致。另以獨立程式重算 AP、配對分數、四種逐 token 機率及區間，最大數值差為 \(3.55\times10^{-15}\)。 [資料準備](contractnli_prepare.py) · [評分程式](contractnli_report.py) · [重現收據](../results/contractnli-evidence-l4/run-verification.json)

正式 L4 讀出處理 131,859 個 token、4,271 個認證停點，12 項前綴檢查通過。正文與命題的前向和擷取約 56.92 秒，整個推論階段約 134.07 秒，GPU 峰值保留約 9.38 GiB。本機評分約 1.04 秒、峰值 114 MiB。第一次工作漏傳子指令，未執行模型前向。兩個執行個體都已釋放，正式工作前後帳戶顯示 734.86 與 734.80 CU。 [正式擷取](../results/contractnli-evidence-l4/extraction.json) · [第一次執行](../results/contractnli-evidence-l4/first-attempt.json) · [正式工作用量](../results/contractnli-evidence-l4/usage.jsonl)
