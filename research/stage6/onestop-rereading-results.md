# OneStop 正文與重讀條件的留出結果

**Qwen 的段末狀態加上重讀條件，未通過預定驗收。** 交互讀出預測答對率隨重讀提高 5.06 個百分點，真人估計提高 4.00 個百分點。差幅方向一致，但估計精度不足以確認落在 ±3 個百分點內。逐題 Brier 也沒有足夠證據優於只用重讀狀態的統計先驗。 [事先固定的比較](onestop-rereading-model.md) · [封存評分](../results/onestop-rereading-l4/scores.json)

封存資料有 1,294 組開發配對及 647 組留出配對。前兩批用於擬合，第三批保留 10 篇文章與 60 位讀者。留出評分使用兩種狀態都有觀測的 216 個段落版本×題號組別，共 1,078 筆作答。Brier 和負對數概似都越低越好，0 為最佳。表中兩種損失按文章等權，重讀差以答對機率的百分點表示。 [材料與分割收據](../results/onestop-rereading-l4/assay.json)

| 讀出 | Brier | NLL（nats） | 重讀減初讀（百分點） |
| --- | ---: | ---: | ---: |
| 共用先驗 | 0.14996 | 0.47702 | 0.00 |
| 重讀先驗 | 0.14956 | 0.47568 | +3.78 |
| Qwen 正文 | 0.15009 | 0.47827 | 約 0 |
| Qwen 正文與重讀加成 | 0.14981 | 0.47693 | +3.80 |
| Qwen 正文與重讀交互 | 0.14967 | 0.47653 | +5.06 |
| 真人 |  |  | +4.00 |

交互讀出減重讀先驗的 Brier 差為 +0.00011，95% 文章配對重抽樣區間 [−0.00231, +0.00250]。交互讀出減加成讀出為 −0.00014 [−0.00061, +0.00031]，減正文讀出為 −0.00042 [−0.00145, +0.00052]。三個區間均跨零。交互讀出減真人的重讀幅度為 +1.06 個百分點，90% 區間 [−1.58, +3.76]，上界超過固定容許差。 [逐項驗收與重抽樣](../results/onestop-rereading-l4/scores.json)

作為診斷，把同一正文前綴固定後只切換重讀狀態。正文讀出的機率完全相同，交互讀出的平均差為 +5.06 個百分點。以篇章分開看，真人的估計差介於 −4.69 和 +12.50 個百分點，交互讀出的預測介於 +0.77 和 +7.83。這 10 篇文章顯示，模型能輸出可隨文字變動的重讀效應，但目前無法證明變動與真人一致。這項讀出保留為實驗結果。 [篇別結果](../results/onestop-rereading-l4/scores.json)

以 Gradio 候選的原始 CUDA 編碼器擷取 69 份去重正文、53,566 個 token、371 個段末狀態。每份正文主擷取一次因果前向。另做 12 次截短或替換後文的控制，最大相對狀態差為 0.00000108。L4 擷取階段 61.6 秒，包含模型初始化、驗證及寫檔。CUDA 峰值配置約 7.19 GiB。回收檔案後確認 L4 已釋放。本機只讀取已擷取的向量擬合，約 0.06 秒，該程序峰值常駐記憶體約 93 MiB。第二次擬合及評分的權重、預測、封存和結果逐位元相同，並另重算所有 Brier 損失。 [段末座標與前綴控制](../results/onestop-rereading-l4/feature-seal.json) · [擬合封存](../results/onestop-rereading-l4/fit-seal.json) · [執行與釋放收據](../results/onestop-rereading-l4/run-verification.json)

依序使用 `research.stage6.onestop_rereading_prepare` 由三份固定來源重建正文與兩組作答，使用 `research.gradio_app.cuda_verification_batch prepare-onestop` 封存正文擷取輸入，再用 `research.stage6.onestop_rereading_fit` 的 `fit` 與 `score` 分開重建預測、評分。欲重算此次評分，可直接使用封存的 [段末狀態](../results/onestop-rereading-l4/features.npz) 或 [預測](../results/onestop-rereading-l4/forecasts.npz)。 [前處理](onestop_rereading_prepare.py) · [擬合及評分入口](onestop_rereading_fit.py) · [原始來源與使用條款](onestop-rereading-audit.md)
