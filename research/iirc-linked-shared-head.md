# IIRC 同一讀出的雙留出驗收

連結正文落點與同文配對各有一份留出。兩份實驗沿用同一個 Qwen3.5-2B 快照及固定投影，但原先的擬合資料不同。官方留出的原始讀出擬合了 3,354 條路徑，其中包括配對留出的 114 條路徑。因此，將來若要在介面使用同一讀出，須用配對實驗從 3,240 條路徑擬合的權重個別通過兩份留出。既有兩份實驗的主要指標、容許差與重抽樣規則維持原樣。 [官方留出規格](iirc-linked-evidence-protocol.md) · [同文配對規格](iirc-identity-holdout.md)

固定配對實驗擬合產生的 `heads.npz`。以原 L4 分片中官方 dev 的 452 條因果路徑重播該讀出。重播依案例和字元停點重新對齊，沿用原始單位向量正規化與訓練資料的幾何尺度。同時重播配對留出的 114 條路徑，逐停點驗證原封存預測與相同權重重播的最大機率差不超過 \(10^{-12}\)。預測封存前只讀兩份留出的文章、錨點、停點、因果狀態及配對實驗的原封存預測。封存同時記錄權重、擬合收據、原始 L4 分片收據、預測及執行規格雜湊。評分時才使用兩份留出標註，且只有兩份都通過各自原定門檻才報告雙留出通過。

以 `iirc_linked_identity replay` 取得配對實驗因果特徵。`iirc_matched_colab prepare` 封裝重播特徵、逐筆訓練區間及固定擬合程式，`run` 在 Colab L4 執行個體內用原定 SciPy CPU 求解器擬合。執行個體只接收這次擬合需要的檔案，無須重新載入 Qwen。

`iirc_linked_shared_head forecast` 接受該實驗的 `--prepared`、`--fit`、`--extraction`，原始資料的 `--original`、`--official-assay`，以及已釋放的原始 L4 `--job`。`iirc_linked_shared_head score` 沿用相同資料、擬合權重及特徵收據，將 `--job` 改成 `--predictions` 指定封存的預測目錄。配對實驗的 `--extraction` 指向重播產生的 `extraction.json`。 [擬合入口](semantic_tracking/iirc_matched_colab.py) · [封存與評分程式](semantic_tracking/iirc_linked_shared_head.py)

這項驗收的端點是具備來源段落、已給錨點和連結正文時，作答者事後標註的文本證據完整出現在哪個區間。兩份留出皆未量測原提問者何時察覺或解開疑問。
