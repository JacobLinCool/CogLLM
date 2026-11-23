# 真人圈選擬合改善 WikiNews 排序，CompLex 的額外增益未確立

在 WikiNews 的 7 個留出文章家族，詞長、詞頻與既有 CompLex 分數的融合讀出，將詞彙排序一致率從 **64.35% 提高至 77.28%**。只用詞長和詞頻擬合也達 **77.19%**。融合相對擬合字面讀出的差為 +0.09 個百分點，配對 95% 區間 [−1.12, +1.21]，逐票 NLL 的差值區間也包含零。兩項證據未達預定的 CompLex 增益門檻。 [預定規格](cwi-lexical-readout-protocol.md) · [結果及區間](../results/cwi-lexical-readout-l4/results.json)

## 材料與比較

沿用先前的 WikiNews 家族切分，排除跨 HIT 重複的句子及無法對應單一完整介面詞的候選。保留 591 個原句、5,723 個候選詞。其中訓練有 338 句、21 個家族及 3,205 個候選，3,204 個完成對齊並用於擬合。開發和測試各有 7 個不同家族。依正規化整句或 25% 五詞片段的標準，訓練與兩組留出及專業新聞均無合格重合。每句直接使用目前 Gradio checkpoint 的一次因果前向及實際顯示精度。凍結原介面分數後，僅使用原訓練組的真人票數擬合三個分數。 [資料類別、排除及分母](../results/cwi-lexical-readout-l4/assay.json) · [凍結權重](../results/cwi-lexical-readout-l4/fit.json)

| 評估材料 | 原介面 | 未擬合的詞長與詞頻 | 擬合詞長與詞頻 | 擬合融合 |
| --- | ---: | ---: | ---: | ---: |
| WikiNews 開發 | 64.91% | 77.39% | 78.82% | 79.22% |
| WikiNews 測試 | 64.35% | 76.48% | 77.19% | 77.28% |
| 專業新聞回溯 | 65.67% | 75.25% | 77.39% | 77.00% |

WikiNews 測試涵蓋 1,358 個候選及 5,224 個真人票數不同的同句詞對。融合相對原介面改善 +12.92 個百分點，區間 [+8.96, +17.10]。逐票 NLL 為 0.2637，擬合字面讀出為 0.2653，差為 −0.00157 nats [−0.00370, +0.00058]。專業新聞回溯上，融合與擬合字面讀出的排序差為 −0.39 個百分點 [−1.55, +0.66]。這個結果限制了既有 CompLex 單一分數的增益主張。直接使用 Qwen 詞末表徵擬合新讀出仍是不同的實驗條件。 [配對統計和覆蓋](../results/cwi-lexical-readout-l4/results.json)

## 重現與資源

用 `cwi_wikinews_prepare` 重建原句與分開保存的三組票數，沿用 Gradio 詞彙匯出器。`cwi_lexical_readout predict` 只讀訓練票數，隨後封存權重、預測及評分程式雜湊，再執行 `score`。封存結果重跑逐位元一致，獨立計算核對所有分母、排序、NLL 及詞彙分數。L4 遠端前向匯出階段為 78.26 秒，配置到核對釋放為 140.32 秒。本機擬合與兩份預測約 0.18 秒，合併重跑最高 RSS 約 174 MiB。 [準備](cwi_wikinews_prepare.py) · [擬合與評分](cwi_lexical_readout.py) · [執行與釋放](../results/cwi-lexical-readout-l4/run-verification.json) · [獨立核對](../results/cwi-lexical-readout-l4/independent-check.json)
