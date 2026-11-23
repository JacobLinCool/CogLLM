# Gradio 詞彙分數未通過專業新聞留出

現行 Gradio checkpoint 在 CWI 2018 專業新聞上的詞彙排序一致率為 **65.67%**。同句的詞長與反詞頻合併基線為 **75.25%**，差值為 −9.58 個百分點，配對 95% 區間 [−14.79, −4.52]。979 個可評候選均對回原文，三項排序門檻未通過。 [預定門檻](cwi-news-holdout.md) · [完整評分](../results/cwi-news-holdout-l4/results.json) · [資料作者論文](https://aclanthology.org/W18-0507/)

## 材料與方法

官方 News 測試分割有 28 個 HIT、177 個原句及 2,095 個候選區段。排除跨 HIT 重複的句子、與先前 CWI 材料相同或達固定片段重合標準的句子，以及未恰好對應單一完整介面詞的候選。以相同規則檢查已掌握的 CompLex train、trial 與 test，沒有發現重合。保留 22 個 HIT、100 個原句及 979 個詞。論文原標註者看過段落，現有發布檔可核對原句和 HIT，未提供可核實的跨句呈現順序。 [來源、座標與重合稽核](../results/cwi-news-holdout-l4/assay.json)

Gradio 候選與已核對的全文介面使用相同的 checkpoint、推論程式和 CompLex 詞末頭。每個原句做一次前向，取得介面小數點後兩位的詞彙分數。真人分數是原研究兩組共 20 位標註者投下的圈選票。在同句的 2,856 對票數不同的詞上比較排序，介面同分得到 0.5 分。先平均同句配對，再平均同一 HIT 的原句，最後以 HIT 等權。 [介面驗證](../results/gradio-combined-overview-l4/run-verification.json) · [預測封存](../results/cwi-news-holdout-l4/seal.json)

## 比較

| 讀出 | 句內排序一致率 | Gradio 相對基線的差，配對 95% 區間 |
| --- | ---: | --- |
| Gradio 詞彙分數 | 65.67% | — |
| 詞長 | 72.50% | −6.83 個百分點 [−13.63, −0.06] |
| 反詞頻 | 67.66% | −1.99 個百分點 [−6.72, +2.47] |
| 詞長與反詞頻合併 | 75.25% | −9.58 個百分點 [−14.79, −4.52] |

全部 2,856 對中，111 對的介面分數相同。補充查核假設這 111 對全部能排對，且其他順位維持原樣，則一致率最多升至 67.34%，仍低於合併基線 7.92 個百分點。在兩個標註者群體分開計算，Gradio 為 66.63% 與 65.12%，合併基線為 75.04% 與 75.61%。此結果限於專業新聞的已評候選及給模型原句的條件，不能估計全文讀者的個人理解強度。 [獨立核對及同分上界](../results/cwi-news-holdout-l4/independent-check.json) · [分組結果](../results/cwi-news-holdout-l4/results.json)

`cwi_news_prepare` 重建原句和分開保存的票數。L4 用 `medreadme_lexical_forecast` 匯出兩套留出共用的 Gradio 詞彙分數，再用 `cwi_news_score` 評分。實際跑完 100 次前向，資源配置到核對釋放約 100 秒。兩次本機評分逐位元相同，各約 0.02 秒，最高 RSS 約 100 MiB。獨立程式重算全部分母、配對效果和區間。 [資料準備](cwi_news_prepare.py) · [評分](cwi_news_score.py) · [執行及釋放收據](../results/cwi-news-holdout-l4/run-verification.json)
