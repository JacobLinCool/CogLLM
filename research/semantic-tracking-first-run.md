# 語意疑問的發起位置、上文指向與後續更新

在既有公開資料中，Qwen3.5-2B 改善讀者首次按鍵回讀的定位，也把更多機率分給真人標出的矛盾上文。解答位置可依同一敘事疑問的標註追到文本行，讀者當下理解的精確 token 仍缺可用標記。 [先前核心功能實驗](core-features-first-run.md) · [時點定義](training-protocol.md#逐-token-的閱讀反應)

## 資料與模型對照

### 讀到哪裡開始回讀

使用 Paape 與 Vasishth 的公開第二實驗。一次閱讀呈現一個實驗預切區段，部分區段只有一個詞。以讀者首次抵達該區段之後、尚未做過往回動作的試次為風險集合，比較下一步按回讀鍵或繼續向前。驗證條件按 64 組材料留出 16 組，兩個實驗版本同組，讀者可以跨材料。比較回讀頻率先驗、位置與長度基線及額外使用 Qwen 狀態的模型。另報當下只顯示一詞的區段。在可觀察到往回的試次，保留首次落點與連續回讀最遠到達區段。這批資料的單詞區段涵蓋按鍵行為，讀者可能在按鍵前已產生困難。原型 stop-and-reread 研究的註冊頁未含逐試次資料。 [原研究第二實驗](https://doi.org/10.5070/G601139) · [實際採用的資料](https://osf.io/j8cbh/) · [另一原型的註冊](https://osf.io/p56j8/) · [事件建置](semantic_tracking/onset_prepare.py)

### 已知矛盾指向哪裡

使用 SCARECROW 中的自相矛盾標註。標註者在全文可見時圈出矛盾片段並圈出相關上文。輸入限於研究公開的模型續寫，排除原新聞引文以及上文指向落在原新聞的試次。保留 1,175 筆可精確對齊的標註，訓練與測試分別 949 與 226 筆，依共同新聞提示與相同續文分組。只提供已圈矛盾的截止位置，評分候選涵蓋矛盾片段之前的詞。讀出比較均勻分布、以距離與詞特徵擬合的基線，以及加上 Qwen 兩處狀態交互特徵的讀出。主要指標是前文圈選區段所獲總機率的負對數，另計最高分詞是否在圈選內。此對照限於已知矛盾發生位置時的指向。 [原研究 §§4.1–4.4](https://aclanthology.org/2022.acl-long.501.pdf#page=5) · [原始資料](https://yao-dou.github.io/scarecrow/) · [資料建置](semantic_tracking/source_prepare.py) · [指向模型](semantic_tracking/source_model.py)

在 Colab L4 使用同一版凍結 Qwen3.5-2B，對每段文字一次 FP32 因果前向，沒有自迴歸生成。保留與原生子詞前綴等同的停點，只用當下和以前的位置表徵。測試標註留在本機，封存機率後合併評分。 [因果前向](core_features/extract.py) · [資料分離](semantic_tracking/prepare.py)

### 同一疑問何時解開

核對 Onea 等人的德語敘事資料。標註團隊先標出同一情節疑問在哪一文本行開啟，再讀一次才標出解答行。各故事的疑問註記支持成對的開啟與結束位置，懸疑強度是另一批參與者的評分。保留結束未確認的疑問作右設限。三篇有原始文本行標註的故事逐篇整篇留出。比較固定的每行結束機率與依疑問已開啟行數分箱的機率。每一折只從其他故事的數值座標估計機率，以真人標註的起點為條件。第四篇的作者處理檔把原位置擴展到詞序，因此只做描述性核對。原始故事與譯本的模型重用權尚未核明，本輪用座標與標註狀態計算。TED-Q 則有較直接的同讀者續評，通常每兩句一次，回答程度達標後停止追蹤。 [Onea 等人 §§3–4](https://doi.org/10.1017/langcog.2025.10057) · [作者資料及標註指南](https://osf.io/m524v/) · [TED-Q 已核對的觀測規則](stage-5-semantic-run.md#問題指向與更新的資料核對)

## 留出結果

### 首次主動回讀

16 組未見材料中有 8,226 次合格停點與 381 次首次回讀。回讀事件的 average precision（越高越好，範圍 0–1）由位置與長度基線的 0.066 升至 0.301，事件機率的 Brier 誤差由 0.04402 降至 0.04183，差 −0.00219，95% 配對材料區間 [−0.00296, −0.00158]。Brier 是預測機率與是否回讀的平均平方差，範圍 0–1，越低越好。單詞顯示區段有 3,313 次合格停點、144 次回讀，average precision 由 0.103 升至 0.248，Brier 差 −0.00142 [−0.00374, −0.00007]。 [首次回讀分數](results/semantic-tracking-first-run/onset-scores.json)

限制在作者操弄的單詞關鍵動詞，236 次停點有 15 次回讀，Brier 差 +0.00005 [−0.00031, +0.00040]。含兩個以上合法較早區段的 335 次回讀，最遠目的地的首選命中率由距離基線的 48.1% 升至 51.0%。起點及最遠目的地都為單詞區段的案例只有七次，兩組各命中零次。原始事件中 303 次首次回讀之後推進超過原起點，按鍵記錄沒有詢問同一疑問是否已釐清。 [關鍵動詞及回讀目的地](results/semantic-tracking-first-run/onset-scores.json)

### 矛盾所指的較早文字

在 226 筆留出指向中，每筆有中位數 59 個候選詞，真人圈選範圍中位數含 6 個候選詞。加入 Qwen 後，真人圈選區段所獲總機率的負對數概似由 **2.077 降至 2.027 nats**，相對距離與詞特徵基線的差為 −0.0496 [−0.0713, −0.0284]。數值越低，表示圈選區段得到越多的機率。最高分詞落在圈選範圍內的比例由 27.4% 升至 31.0%，差 +3.54 個百分點，區間 [−1.71, +8.76]。目前有排序訊號，精準首選還有較大誤差。 [上文指向分數](results/semantic-tracking-first-run/source-scores.json)

### 故事情節疑問的解答位置

三篇行級資料有 196 個標註問題，其中 183 個有解答行。逐篇留出的配對資料涵蓋 14,945 個待解問題 × 行。用已開啟多久估計結束機率，三篇等權的平均負對數概似從固定機率的 0.07410 降到 0.07242 nats（越低越好），average precision 由 0.0140 升到 0.0239（越高越好）。負對數概似在兩篇下降、一篇上升。這是已知問題起點的時間基線，沒有文章語意輸入。 [逐篇結果及重跑數據](results/semantic-tracking-resolution/narrative-metadata.json) · [計算程式](semantic_tracking/resolution_prepare.py)

合計四篇資料中，115 個有一個待解問題被標為已解、同時還有其他問題跨過該位置仍未解的標註時點，有 79 個無法由嚴格後進先出規則重現，判定已允許同時開啟並列。結果說明這些敘事標註允許較早問題先結束。這些解答位置由重讀後標註，分辨率受原始文本行限制。即使有逐詞懸疑評分，也無法由該評分確定讀者在哪個 token 注意到答案。 [問題組合、設限與配對統計](results/semantic-tracking-resolution/narrative-metadata.json)

## 執行與重現

NVIDIA L4 以固定 Qwen3.5-2B revision 擷取 654 段文字、71,534 個非填充 token 和 51,770 個原文停止位置，沒有 tokenizer 對齊排除。12 個前綴檢查通過，前向及取值 30.31 秒，來源讀出和回讀讀出的程式內擬合與準備分別 5.33 和 3.53 秒。擷取峰值保留 7.63 GiB GPU 記憶體。從配置確認至停止請求為 167.23 秒，L4 已確認釋放。帳戶同時有其他使用中的指派，帳戶差額不能單獨歸給本次工作。 [擷取與前綴](results/semantic-tracking-first-run/extraction.json) · [Colab 紀錄](results/semantic-tracking-first-run/manifest.json) · [用量](results/semantic-tracking-first-run/usage.jsonl)

依序重跑 `onset_prepare`、`source_prepare` 和 `prepare`，再於 L4 使用 `core_features.extract`、`source_model` 及 `onset_model`。留出預測經雜湊封存後，以 `onset_score` 和 `source_score` 在本機接上標註。兩套留出評分逐位元重跑一致，主要誤差及首選命中已另行重算。敘事基線用 `resolution_prepare` 讀取作者發布的標註檔，輸出不含原文。 [輸入雜湊](results/semantic-tracking-first-run/payload-audit.json) · [回讀核對](results/semantic-tracking-first-run/onset-independent-verification.json) · [指向核對](results/semantic-tracking-first-run/source-verification.json) · [結果檔完整性](results/semantic-tracking-first-run/verification.json)
