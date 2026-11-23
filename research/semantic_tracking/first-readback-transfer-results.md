# 只輸入正文的首次回讀留出結果

英德合併訓練的一組讀出加入 Qwen 狀態後，在兩份未見材料上都比位置基線更準確地估計首次主動回讀。兩份資料的 Brier 差異區間均低於零。德文平均機率通過事先固定的 ±1 個百分點校準門檻，英文區間的上界為 +1.06 個百分點。整體未通過兩份資料皆須達標的規則。 [評分及預定門檻](../results/first-readback-transfer-l4/scores.json) · [事先固定的方案](first-readback-transfer.md)

| 留出材料 | 按鍵回讀／合格停點 | 位置 Brier | Qwen＋位置 Brier | 差值與 95% 材料家族區間 | 平均機率校準差與 90% 區間 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 英文，5 組 | 114／2,717 | 0.03963 | 0.03758 | −0.00205 [−0.00315, −0.00084] | +0.51 [−0.09, +1.06] 個百分點 |
| 德文，16 組 | 381／8,226 | 0.04417 | 0.04188 | −0.00229 [−0.00307, −0.00167] | +0.05 [−0.47, +0.60] 個百分點 |

Brier 是每次預測機率與二元按鍵觀測的平均平方誤差，越低越好。校準差是平均預測機率減真人按鍵率。合併讀出以兩研究等權擬合，在各研究內按原材料家族重抽樣 10,000 次，讀者固定。原實驗以按鍵逐步顯示區段，評分對象是第一次向前閱讀時，下一步按回讀鍵或向前。兩研究各自沿用原家族留出，讀者可跨材料。538 個已設計留出停點均先取得預測，真人資料在其中 534 處有合格觀測。 [資料稽核與分母](../results/first-readback-transfer-l4/assay.json) · [英文原研究](https://doi.org/10.5070/g6011182) · [德文原研究](https://doi.org/10.5070/G601139)

在一次只顯示一個詞時，德文的 Brier 差為 −0.00161，95% 區間 [−0.00393, −0.00023]，涵蓋 3,313 次機會與 144 次回讀。英文 867 次機會中只有 15 次回讀，差為 −0.00004 [−0.00013, +0.00005]。只用英文訓練時，德文的平均預測機率為 11.34%，真人按鍵率為 4.63%。研究內與跨研究的讀出有不同程度的機率偏差。現有證據支持這兩組材料的顯示停點，按鍵前可能已出現困惑，區段也可能包含多個詞。一般文章中的疑問起始與解除仍待直接觀測驗證。 [按粒度與訓練範圍的完整分數](../results/first-readback-transfer-l4/scores.json)

L4 使用已驗證 Gradio 候選的凍結 Qwen3.5-2B FP32 骨幹。168 段材料共 8,163 tokens，全部 2,344 個原顯示停點通過前綴認證，每段一次模型前向，12 項因果檢查通過。以新讀出擬合並輸出預測的遠端階段共 44.2 秒，骨幹前向與擷取為 3.69 秒，實例已核對釋放。取得封存的預測後才在本機接上留出觀測。評分重跑為 0.54 秒，最高 RSS 為 47.2 MiB。36 組指標另外以展開的逐次觀測核對。此讀出保留在實驗結果。 [擷取紀錄](../results/first-readback-transfer-l4/extraction.json) · [因果檢查](../results/first-readback-transfer-l4/causality.json) · [預測封存](../results/first-readback-transfer-l4/seal.json) · [驗證與釋放](../results/first-readback-transfer-l4/run-verification.json)

本機重跑只需 NumPy、[準備好的公開資料](first_readback_prepare.py)和[已封存的預測](../results/first-readback-transfer-l4/predictions.jsonl)。

```sh
python3 -m research.semantic_tracking.first_readback_score \
  --assay-root .cache/cogllm/semantic-tracking/first-readback-v1 \
  --predictions-dir research/results/first-readback-transfer-l4 \
  --output /tmp/cogllm-first-readback-replay.json
```
