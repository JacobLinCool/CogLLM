# 三骨幹在真人閱讀資料上的留出比較

2026-11-21。固定既有真人資料、留出群組與讀出規格後，詞彙難度的留出誤差增益在三個骨幹都重現。句級理解自評的增益由 Qwen 和 LFM3 重現。三者都低估白話改寫的真人收益，也都未重現矛盾前文對真人首次回報機率的幅度。 [固定比較規格](protocol.md)

## 真人端點與共同基線

| 留出端點 | 共同基線 | Qwen3.5-2B | LFM3-1.7B-Base | Gemma 4 E2B |
| --- | ---: | ---: | ---: | ---: |
| CompLex 目標詞難度 MAE | 0.10959 | 0.08064 | 0.08558 | 0.08716 |
| PLABA-EVAL 句級閱讀容易度 NLL | 1.34534 | 1.33344 | 1.32789 | 1.34361 |
| PLABA-EVAL 句級理解自評 NLL | 1.34540 | 1.31051 | 1.31057 | 1.34543 |
| Frick 整篇首次回報矛盾 NLL | 1.73313 | 1.73022 | 1.73148 | 1.73206 |
| Frick 目標句首次回報矛盾增幅，真人 +48.27 個百分點 | +0.0393 | +0.0449 | +0.0386 | +0.0390 |

誤差越低越好。PLABA-EVAL 的共同基線使用句長與已知改寫版本。CompLex 和 Frick 使用各自固定的表面特徵基線。Frick 表格使用三骨幹共用的句間換行後停點。後文的區間是既有留出預測的 95% 群組重抽樣區間。PLABA-EVAL 按讀者及材料的 25 個連通群組，CompLex 按 1,066 個測試句群組，Frick 同時重抽故事及讀者。 [CompLex 分數](../results/backbone-transfer-l4/lexical-sentence/complex/metrics.json) · [PLABA-EVAL 分數](../results/backbone-transfer-l4/lexical-sentence/plaba/metrics.json) · [Frick 分數](../results/backbone-transfer-l4/frick/separator_end/results.json)

### 目標詞難度

三組在共同的 8,666 筆訓練和 1,093 筆測試上各自擬合相同 ridge 讀出。三組相對表面基線的 MAE 改善區間均排除零。LFM3 和 Gemma 相對 Qwen 的誤差分別增加 0.00494（區間 0.00238 至 0.00746）和 0.00652（0.00323 至 0.00970）。分詞檢查在原分割排除一筆訓練與一筆測試，因 LFM3 的 token 涵蓋範圍越過目標詞。CompLex 的真人難度評分是在看完整句後取得。此處模型讀至指定詞末，量測的是當時表徵能否讀出該評分訊號。 [原始切分和讀者所見文本](../stage-5-complex-run.md) · [共同樣本與配對差](../results/backbone-transfer-l4/lexical-sentence/complex/metrics.json)

同一類型與詞形在不同句子中的排序共有 1,050 對。表面基線一致率為 51.0%，Qwen 為 50.9%，LFM3 為 50.1%，Gemma 為 51.8%。這些點估計接近 50%，目前無法從總體 MAE 的增益推論模型能按上下文排出同一詞更難的用法。排序稽核沿用了 Stage 5 的句群排除與同分規則，去除了一組同句配對。封存預測與 MAE 均一致。 [排序稽核與重播](../results/backbone-transfer-l4/run-verification.json)

### 句級評分與改寫

609 句與 1,826 筆雙量尺評分全數保留。理解自評上，LFM3 減 Qwen 的 NLL 差為 +0.00007 nats（區間 −0.00591 至 +0.00669）。兩者均優於長度加已知版本基線。Gemma 的理解自評 NLL 較 Qwen 高 0.03492（+0.01867 至 +0.05205），與共同基線的差值區間跨零。閱讀容易度上，LFM3 的 NLL 點估計最低，與 Qwen 的差值區間跨零。以排序機率分數衡量容易度，Qwen 和 LFM3 對共同基線的改善區間均排除零，Gemma 區間跨零。 [五折留出與兩種評分指標](../results/backbone-transfer-l4/lexical-sentence/plaba/metrics.json)

265 組句對上，真人的白話版本比原版平均提高 0.537 分閱讀容易度和 0.723 分理解自評。相應的模型差幅為 Qwen +0.277／+0.341、LFM3 +0.288／+0.350、Gemma +0.190／+0.209。三組在兩個量尺的模型減真人差值區間均低於零。已知版本的基線可以直接利用改寫條件，平均差幅為 +0.496／+0.704，說明重現平均差幅與準確預測個別句子的分布是兩項分開的驗證。 [改寫差幅與群組區間](../results/backbone-transfer-l4/lexical-sentence/plaba/metrics.json)

### 語意矛盾回報

20 組故事、180 位讀者、1,800 條試次使用原故事家族留出。真人在矛盾目標句首次回報的機率增加 48.27 個百分點（39.10 至 57.26）。三骨幹的點估計均約 +0.04，與表面基線同量級，全部落在預定的 ±10 個百分點容許差之外。Qwen 和 LFM3 的整篇 NLL 較表面基線略低，Gemma 的改善區間跨零。考慮所有句子的整篇誤差可以很小，仍可能漏掉真人在某個特定句子的大幅反應。 [共同停點的首次回報與當句回報](../results/backbone-transfer-l4/frick/separator_end/results.json)

LFM3 把 584 個非末句的句末和原有兩個換行編在同一 token，無法對齊那些原始字元停點。三模型的主比較統一移至原換行後，未讀入下一句的任何非空白字元。保留原始停點時，Qwen 和 Gemma 的整篇 NLL 均略優於表面基線，兩者仍遠低估目標效應。這個結果只驗證受提示的逐句首次回報。現有分數無法驗證真人首次卡住的 token、疑問所指的上文，或同一疑問解除的位置。 [原始句末副對照](../results/backbone-transfer-l4/frick/unicode_end/results.json) · [分詞比較](../results/backbone-transfer-l4/frick/results.json)

## 已驗證的執行範圍

三骨幹都以凍結的官方文字權重擷取最後一層狀態，使用正文且無讀者背景條件。每骨幹在 Colab L4 以 FP32 各讀 9,284 份正文，每份只做一次因果前向。10 項獨立截短及後文替換驗收均通過，每份正文的觀測停點一次取齊，全程無自回歸解碼。讀出頭在 CPU 按原訓練折各自重擬合。此輪未比較注意力、中間 logits、骨幹微調或聯合訓練的效果。留出沿用既有公開材料，並保留每個骨幹的原生 tokenizer 與隱藏寬度，所以差異反映整組模型配置的表現。 [執行驗證](../results/backbone-transfer-l4/run-verification.json)

| 骨幹 | 正文前向與表徵擷取 | 峰值 GPU 保留 | L4 配置到確認釋放 |
| --- | ---: | ---: | ---: |
| Qwen3.5-2B | 183.9 秒 | 7.36 GiB | 341 秒 |
| LFM3-1.7B-Base | 175.4 秒 | 6.71 GiB | 340 秒 |
| Gemma 4 E2B | 214.4 秒 | 17.97 GiB | 553 秒 |

三張 L4 均已確認釋放。速度及顯存為本次固定資料與批次的單次測量。Gemma 使用原始 checkpoint 中的文字子模型。LFM3 使用相容層保留原廠層順序與 RoPE 參數，官方檔案與權重維持原有雜湊。以封存表徵重播評分，輸出雜湊逐檔一致。Qwen 舊有表徵在同組重新擬合後，CompLex 的 MAE 與本次相差約 0.000008，PLABA-EVAL 的兩項 NLL 分別相差約 0.000019 與 0.0000002。 [執行及重播收據](../results/backbone-transfer-l4/run-verification.json) · [Qwen 封存對照](../results/backbone-transfer-l4/qwen-replay.json) · [LFM3 載入細節](lfm3-source.md)
