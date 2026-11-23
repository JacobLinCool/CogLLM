# Stage 6 RACE 換語料驗證

用 OneStopQA 的開卷真人作答訓練讀出，固定預測後，在同一個 Prolific 實驗中換成 RACE 的文章與題目。兩個語料的正文、題幹與四個選項同時可見，並共用 970 個讀者鍵。這項實驗驗證未用於讀出擬合的材料。估計的反應是讀者選了哪個顯示選項。 [Stage 6 計畫](stage-6-holdout-plan.md) · [原實驗 §4.3](https://aclanthology.org/2020.acl-main.507.pdf#page=6) · [材料和讀者重疊稽核](results/stage6-readiness/race-audit.json)

## 評分前固定的做法

RACE 使用作者釋出的 Prolific TSV 與實際顯示的正文、題幹和選項順序，一共 216 篇、648 題和 1,940 筆作答。依之前 OneStopQA 的同一個提示格式建置兩種可見條件。分開去重各得 1,851 種輸入。模型讀到完整提示的最後一個 token 後取隱藏狀態。固定 Qwen3.5-2B revision、Transformers 5.17.0 和 FP32 因果前向，每個輸入各通過一次。GPU 輸入只包含提示 ID 及文字，讀者背景留空。正誤評分所需的答案鍵逐題比對固定版原 RACE，正文則採 Prolific 的實際呈現版本。研究用途依 RACE 原發布的非商業研究條件。 [作者資料](https://github.com/berzak/onestop-qa/tree/7a058755e1ff0d3f835a67109c669c7e294d4239/human_experiments) · [原 RACE 版本](https://huggingface.co/datasets/ehovy/race/tree/2fec9fd81f1dc971569a9b729c43f2f0e6436637) · [使用條件](https://www.cs.cmu.edu/~glai1/data/race/) · [既有 L4 特徵程式](joint/onestop_features.py)

用原 OneStopQA 設計和已保存的 L4 特徵在本機 CPU 重建五個折的線性頭。保持同一個 L2 懲罰、特徵正規化、截距與求解方式，先核對重建的 OneStopQA 留出機率。顯示位置先驗在各訓練折按真人所選的位置計算。每個條件對 RACE 的預測是五折各自的機率取算術平均。依每次實驗的顯示位置輸出四個機率，先封存預測及雜湊，再在本機接上真人作答。 [原 OneStopQA 結果](joint-first-run.md#onestopqa-留出結果) · [原讀出程式](joint/onestop_probe.py)

主要指標是所選選項的負對數概似（NLL），以 nats 計，越低代表分配給真人實際所選的選項越高的機率。主要差值為有正文減只有題目與選項。多類 Brier 是四個選項的平方機率誤差和，範圍 0–2。正誤 Brier 比較給正解的機率和真人是否答對，範圍 0–1。兩者越低越好。每個試次等權，配對 95% 區間用種子 43 重抽樣 10,000 次，分別對文章和讀者重抽，將兩種抽樣重數相乘得到每個試次的權重。模型權重固定。另按 RACE 中學、高中題庫和真人答對或答錯做描述比較。 [留出評分程式](stage6/race_score.py)

## 留出結果

| 條件 | 真人所選選項 NLL（nats） | 四選項 Brier | 正誤 Brier |
| --- | ---: | ---: | ---: |
| OneStopQA 顯示位置先驗 | 1.3849 | 0.7493 | 0.4521 |
| 題目與選項 | 1.3519 | 0.7323 | 0.4395 |
| 正文、題目與選項 | 1.0686 | 0.5659 | 0.3169 |

加入正文的主要配對 NLL 差為 −0.2833 nats，95% 區間為 [−0.3159, −0.2511]。四選項 Brier 差為 −0.1664 [−0.1860, −0.1470]，正誤 Brier 差為 −0.1226 [−0.1371, −0.1080]。在固定 OneStopQA 讀出的條件下，正文也能改善新語料上真人選答的機率分配。 [完整分數與區間](results/stage6-race/scores.json)

按真人作答結果分組觀察，1,514 筆答對試次的 NLL 由 1.3405 降至 0.9628，426 筆答錯試次的 NLL 由 1.3926 升至 1.4446。加入正文帶來的整體改善來自答對試次，同時降低了真人實際錯選的選項的幾何平均機率。這是按已發生的反應分組的描述結果。測試的兩批材料共用讀者，觀測也發生在開卷作答階段。閱讀中疑問首次出現、跨句累積與回讀的定位，需用在那些位置記錄的人類反應評估。 [作答分組結果](results/stage6-race/scores.json)

## 執行與重現

Colab NVIDIA L4 以固定 Qwen revision 擷取 3,702 個提示的 800,720 個非填充 token。前向與特徵讀取合計 321.22 秒，GPU 記憶體保留量峰值為 9.29 GiB。12 項前綴一致性檢查通過。從 runtime 配置確認到發出停止要求為 407.19 秒，下載產物通過雜湊驗證，L4 已釋放。OneStopQA 讀出重建、RACE 機率計算及真人評分在本機 CPU 完成。 [Runtime 紀錄](results/stage6-race/manifest.json) · [擷取時間](results/stage6-race/extract/timings.json) · [前綴檢查](results/stage6-race/extract/causality.json)

OneStopQA 五折留出機率與歷史 L4 紀錄的最大絕對差為 \(8.07\times10^{-7}\)。RACE 預測在讀取其真人選答之前封存。另以原始特徵與五折權重獨立重算機率，最大絕對差為 \(1.67\times10^{-16}\)。評分程式重跑的輸出逐位元一致。重現順序為建置無作答的提示、擷取凍結特徵、重建並驗證舊讀出、封存 RACE 預測、對齊 RACE 作答，最後評分。 [提示及作答對齊](stage6/race_prepare.py) · [重建及封存](stage6/race_predict.py) · [OneStopQA 重建紀錄](results/stage6-race/onestop-rebuild.json) · [RACE 獨立核對](results/stage6-race/independent-verification.json)
