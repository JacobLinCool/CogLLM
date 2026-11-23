# IIRC 連結文件的證據到達驗收

這項實驗檢驗一個已知疑問錨點是否能幫助定位連結文章中的所選證據。IIRC 的提問者先讀原始段落並指出連結，作答者之後取用連結文章並標註支持文字。資料的可測端點是已標註的證據全文在第幾個讀取停點出現。兩個任務由不同參與者完成，原提問者解開疑問的時間未記錄。 [IIRC §§2.2–2.3](https://aclanthology.org/2020.emnlp-main.86.pdf#page=4)

## 材料與推論輸入

只保留原始段落中唯一可定位的單一連結。該連結在作者發布的同一段落中只對應一個人寫下的問題，作答者提交了答案，且標記了一個可精確對到同一篇連結文章的外部證據區段。座標取自 `context.indices`，並逐字核對作者發布的去除 HTML 標籤後原文。未作答、沒有外部支持、使用多個區段及對齊失敗分別計數。 [作者文章解析器](https://github.com/jferguson144/IIRC-baseline/blob/main/util.py#L35-L44) · [輸入建置程式](semantic_tracking/iirc_linked_prepare.py)

用原始段落、兩個換行及連結文章從字元零起的最多 8,000 個 Unicode 字元構成一條路徑，一次因果前向。原始段落中的已提供錨點、連結正文起點和因果停點是模型可用的定位資訊。從連結起點每 128 字元設一個名目停點，再回移至可用獨立重分詞驗證的邊界。結尾和錨點句內終點也須對齊。所有停點僅從輸入文字與已給定的錨點建置。原始問題字串、答案、支持區段及連結標題保留於推論資料之外。 [輸入建置程式](semantic_tracking/iirc_linked_prepare.py)

官方 dev 文章用於留出。訓練資料剔除原始或連結文章識別名、精確源文字雜湊與 dev 原始段落或可見連結相交的個案。統計重抽樣群組為由原始及連結文章名稱與源文字雜湊連成的文章組。訓練保留 3,354 條路徑，從 2,352 個文章組取得 3,174 個已觀測區間及 180 個右設限。留出保留 452 條路徑，從 269 個文章組取得 413 個已觀測區間及 39 個右設限。切分間的文章名和精確原文字雜湊均無重疊。 [切分與來源稽核](results/iirc-linked-prepare-audit.json)

## 指標與驗收門檻

令 \(T\) 為第一個讀到全部標註證據的連結正文停點，編號從 1 起算。若證據結尾超過最末停點，該案例右設限於最末停點 \(L\)。設限包括已讀到證據一部分的情形。已觀測事件計 \(-\log P(T=t)\)，設限事件計 \(-\log P(T>L)\)。負對數似然以 nats 計，每個問題等權，越低越好。另以各問題在自身連結停點的累計事件機率均方誤差計 Brier 分數，每個問題等權。Brier 分數介於 0 和 1，越低越好。

同一資料、切分、事件目標、學習目標及評估停點使用三類讀出。位置基線使用已可見字元長度、錨點範圍及當下停點寬度等位置特徵。第二個基線加入同一路徑的連結正文因果狀態，省略顯式錨點向量。候選讀出再加入已給定錨點狀態與連結停點狀態的交互。每個停點使用的特徵限於當時已呈現的文字和位置，累計機率由每一步的條件事件機率推得。

預定門檻以 10,000 次成組配對 bootstrap 的雙尾 95% 區間評估。候選減位置基線的平均負對數似然差上界須低於或等於 −0.02 nats，Brier 分數差上界須低於或等於 +0.01。支持顯式錨點帶來額外資訊的主張，另要求候選減第二基線的負對數似然差上界小於零。另報告最末停點的平均校準差及機率最高區間或設限類別的命中率。連結正文狀態來自同一條因果路徑，可能已保留原始段落語境。這項比較衡量顯式錨點向量的增量。嚴格的同一連結文章、相同停點、互斥支持交換有 37 組訓練對，dev 有 0 組，因此本切分無法評估更換錨點是否會讓模型轉而定位另一段證據。 [評分程式](semantic_tracking/iirc_linked_score.py) · [切分稽核](results/iirc-linked-prepare-audit.json)

## 執行與封存

資料建置已完成。先用 `iirc_linked_pack` 建立子目錄並封存六個模型端輸入成 `linked-evidence-model-inputs.tar.gz`。封包包含兩個切分的純文字路徑與座標、訓練區間和 `assay.json`。留出標註保存在本機準備目錄，封包僅記錄其雜湊。將封包與 repo 的 `research` Python 模組放入 L4 環境並解壓，依序執行以下指令。 [輸入封裝](semantic_tracking/iirc_linked_pack.py) · [一次因果前向擷取](semantic_tracking/iirc_linked_extract.py) · [讀出擬合](semantic_tracking/iirc_linked_fit.py)

```sh
python -m research.semantic_tracking.iirc_linked_extract --inputs inputs --output extraction
python -m research.semantic_tracking.iirc_linked_fit --inputs inputs --features extraction/features.npz --extraction extraction/extraction.json --output fit
```

擷取以 24 條路徑和 6,144 個填補後 token 為每批上限，將 191,753 個已驗證停點各投影為 128 維 FP32 單位向量。每條路徑只在正式擷取進行一次主幹前向。GPU 前綴檢查另用抽樣路徑驗證截短、右填補和替換後文。可在相同執行環境與輸出目錄續跑已封存的分片。用先前 IIRC 短段落實測的 2,419 token/s 粗估，本次 6,704,880 token 需約 46 分鐘前向及讀出。路徑較長，首個分片以前尚無本實驗實測速度。 [前次 L4 量測](results/iirc-anchor-run/extraction.json)

取得留出預測與 `fit-manifest.json` 後，在本機從準備目錄接上 `dev/locked-labels.jsonl` 與 `dev/locked-matched-swaps.jsonl` 執行 `iirc_linked_score`。評分入口會驗證封存雜湊後才讀入留出觀測值。 [留出評分](semantic_tracking/iirc_linked_score.py)

## 遠端執行紀錄

首個 L4 封包未含一項間接匯入的模組，已補齊並加入配置前的相依檢查。第二次執行通過 L4 前綴檢查，首個分片處理 276,318 token、耗時 119.36 秒，PyTorch 顯示 GPU 記憶體保留峰值 8.79 GiB。最後可取回的進度是 29 個分片中的 12 個，共 3,223,641 token。之後 Colab 伺服器清單已無該執行個體，未取回分片特徵及留出預測。下一次將在每個分片完成時下載並驗證，避免執行個體消失時一併失去擷取結果。 [執行程式](semantic_tracking/iirc_linked_colab.py)

加上耐久下載後的首次工作階段仍因遠端停止而結束，已保存 9 個驗證分片。同一輸入、原始封包和因果檢查收據上的續跑已核對新分片。2026-11-21 07:18（台北時間）已保存 13／29 個分片，尚未完成留出預測及評分。前 11 個分片的實測擷取中位數為 116 秒。 [續跑檢查與下載規則](semantic_tracking/iirc_linked_colab.py)

## 官方留出結果

從 20 個已驗證分片續跑，在單張 NVIDIA L4 上取回全數 29 個分片。整個擷取累計 6,704,880 個 token 及 191,753 個因果停點，前向與讀出累計 2,820 秒，平均每秒 2,378 token。分片的 PyTorch 記憶體保留峰值為 13.84 GiB。最後一段續跑約 18 分鐘，含部署、擷取及讀出擬合。已核對輸出、收據與 L4 釋放。 [特徵與停點收據](results/iirc-linked-evidence-l4/extraction.json) · [當次執行紀錄](results/iirc-linked-evidence-l4/run-verification.json)

官方留出共 452 條路徑，413 個已觀測區間、39 個右設限、269 個文章家族。表中 NLL 和累計 Brier 採各路徑等權。 [封存評分](results/iirc-linked-evidence-l4/scores.json)

| 讀出 | 區間 NLL（nats） | 累計 Brier |
| --- | ---: | ---: |
| 位置 | 2.65950 | 0.15146 |
| 同路徑正文狀態 | 2.62911 | 0.14977 |
| 顯式錨點與同路徑正文 | 2.62312 | 0.14917 |

顯式錨點相對位置的 NLL 差為 −0.03638，95% 區間 [−0.05431, −0.01899]。區間上界略高於預定的 −0.02 門檻。累計 Brier 通過非劣門檻，差值為 −0.00229，95% 區間 [−0.00437, −0.00037]。顯式錨點相對同路徑正文的 NLL 差為 −0.00599，95% 區間 [−0.01590, +0.00384]。三項門檻通過一項，現有官方留出證據不足以讓這組權重進入 Gradio 的證據抵達功能。後續依原規格用配對資料擬合的一組權重進行雙留出驗收。 [同一讀出雙留出](iirc-linked-shared-head.md)
