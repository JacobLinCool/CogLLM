# PLABA-EVAL 明確指認片語的跨語料診斷

本實驗檢查從 IIRC 擬合並固定的片語讀出，能否在作者提供的 PLABA-EVAL 句界內找回讀者明確指出的困難片語。第一輪讀者評分時可看全文。自由回饋中有 89 筆通過人工語義查核和逐字唯一對齊。合併同一正文的相同半開字元區段後，共有 94 個目標，來自 70 句、19 個文章家族與 12 個讀者連通群組。 [PLABA-EVAL 研究](https://aclanthology.org/2026.readi-1.4/) · [來源及區段稽核](results/plaba-explicit-span-audit.json)

PLABA-EVAL 的正文、讀者和句級容易度及理解自評已用於 CogLLM 開發。結構化疑難回覆亦用於句級疑難讀出，開卷作答曾用於資料及理解端點的開發。一些正文也已有介面展示。這次只載入原 IIRC 的兩組權重及固定投影。讀出無 PLABA 擬合、閾值調整或介面選句。這是既有 PLABA 材料上未用於擬合的自由回饋端點，結論限於跨語料診斷。 [既有 PLABA 實驗](stage-2-data.md) · [句級讀出](stage-3-first-run.md) · [疑難勾選](stage-3-feature-ablation.md) · [介面驗證](gradio_app/README.md) · [原 IIRC 實驗](iirc-anchor-run.md)

## 一次前向與位置對照

遠端輸入包含第一輪全部 40 篇純正文的 323 個作者句界，涵蓋 8,273 個 Qwen token。由於全數送入，遠端從資料選取也無法得知哪些句子得到陽性回饋。人工審查程式內含指認詞語，因此只使用明列的遠端原始碼封包。原始回饋、局部評分、指認區段與審查程式留在本機。 [轉移實作](semantic_tracking/plaba_span_transfer.py)

凍結 Qwen3.5-2B，對每篇文件執行一次 FP32 因果前向。兩組共享原 IIRC 的一至八個連續詞段候選及非重疊擇優規則。對每句只保留完全位於該句的候選。兩組的形式特徵從讀到該句末的原文前綴計算。Qwen 組另使用相同前綴中、候選片語首尾詞段的因果向量。保留作者句末的位置必須逐一符合獨立前綴分詞的 token 和偏移。原文擷取程式額外抽樣檢查截短、右填補及替換後文。 [固定 IIRC 候選及權重](semantic_tracking/iirc_anchor_fit.py) · [因果檢查](core_features/extract.py)

## 封存與評分

分母為全部 94 個不重複的陽性區段。每個區段獨立看前一名與前五名中是否至少有一個候選達字元交並比 ≥0.5，並另檢查完全相同的起訖位置。四個命中率都以比例表示，越高代表找回越多明確指認的困難區段。主要比較固定為前五名重疊命中率。對兩組差值做 10,000 次配對重抽樣，同一讀者連通文章組內的區段一併重抽。報告兩組絕對值、差值、95% 區間及候選空間可達上限。12 組材料限制區間精度。句界是已知輸入，未獲指認的文字缺少可用的負例標註。回饋中的命名位置沒有對應閱讀中反應的時間或強度。

將無標註封包解至 `inputs`，原始碼封包解至執行目錄。L4 只執行預測，保存預測、封存清單及擷取和因果檢查檔。取得封存後才在本機執行第二條指令。 [獨立評分入口](semantic_tracking/plaba_span_transfer.py)

```sh
python -m research.semantic_tracking.plaba_span_transfer forecast --inputs inputs --output transfer
python -m research.semantic_tracking.plaba_span_transfer score --inputs inputs --labels research/results/plaba-explicit-span-audit.json --predictions transfer/predictions.jsonl --seal transfer/forecast-manifest.json --output scores.json
```

無標註資料封包的 SHA-256 為 `8d1f8ca85cc6ec5ec3afa3a3afbf95638a29a003fc08b00b29d39717fe22c306`。純推論原始碼封包 SHA-256 為 `6f393059baa58886af56154c7aa0d60c04e674ce9175866df6607ed38a5201b3`。所有作者句界已經用固定 Qwen 分詞器核對。純 CPU 合成測試驗證 Unicode 後文剪除與替換時的排序一致、重複目標合併、陽性召回及封存驗證。 [輸入建置與封存](semantic_tracking/plaba_span_transfer.py)

## Colab L4 轉移結果

以真人指認的 94 個區段為分母，凍結 IIRC 的 Qwen 讀出，在前五個候選找回 52 個達到交並比 0.5 的區段。形式基線找回 19 個，差 +35.11 個百分點，讀者連通群組重抽樣的 95% 區間 [+20.83, +50.00]。精確邊界為 31／94 對 9／94。前一個候選完全命中時兩組同為 4／94。結果支持在已知句界找出多個候選片語。這批回饋沒有負例或首次發生時間。 [獨立評分](results/plaba-span-transfer-l4/scores.json) · [封存預測](results/plaba-span-transfer-l4/forecast-manifest.json)
