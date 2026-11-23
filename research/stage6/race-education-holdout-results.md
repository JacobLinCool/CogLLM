# 英文開卷作答的教育程度留出結果

加入教育程度後，所選選項的平均 NLL 從 0.89452 降為 0.89285 nats。改善量為 **+0.00167 nats**，讀者 × 文章配對重抽的 95% 區間為 [−0.00411, +0.00718]，未達事先的改善門檻。這次使用 970 位讀者對 216 篇 RACE 材料的 1,940 次開卷作答。每折只用其他讀者的 OneStopQA 選答來擬合。 [固定規格](race-education-holdout-protocol.md) · [完整分數](../results/race-education-holdout/scores.json)

| 讀出 | 所選選項 NLL（nats） | 正誤 Brier |
| --- | ---: | ---: |
| 表面特徵 | 1.37645 | 0.43650 |
| 正文、題目與選項 | 1.06891 | 0.31710 |
| 空背景，校準選擇機率 | 0.89452 | 0.23006 |
| 教育程度，位置偏移 | 0.89470 | 0.23014 |
| 教育程度，位置偏移與 logits 斜率 | 0.89285 | 0.22821 |

兩個分數皆越低越好。每折的空背景校準使用 OneStopQA 折外 logits，沒有使用測試 RACE 的反應。只取 938 位有教育程度紀錄的讀者，改善量為 +0.00173 nats，95% 區間 [−0.00405, +0.00746]。32 位背景空白讀者的 64 筆試次中，條件讀出與空背景分布逐位元相同。 [完整分數](../results/race-education-holdout/scores.json)

本次比較描述既有教育程度與英文開卷選答之間的關係。讀者仍來自同一項 Prolific 研究，訓練與評估在每折使用不重疊的讀者鍵與兩套不同材料。資料未測各篇的概念先備知識，也未在閱讀中逐句量測負荷。 [原研究 §4.3](https://aclanthology.org/2020.acl-main.507.pdf#page=6) · [來源及分割收據](../results/race-education-holdout/forecast-seal.json)

沿用既有 L4 的 FP32 因果前向特徵。原封存 RACE 機率逐位元重播一致，所有新預測先封存再接上選答。以實際選項直接索引封存機率重算逐次 NLL，與原評分函式完全一致。兩次 CPU 執行的彙總檔逐位元相同，共用時 8.30 秒，程序峰值記憶體約 273 MiB。 [重現程式](race_education_holdout.py) · [封存收據](../results/race-education-holdout/forecast-seal.json)
