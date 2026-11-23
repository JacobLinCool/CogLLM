# Anglin 的首日信念轉移結果

**CogLLM 的 Anglin 轉移未通過本次驗收。** 六項比較有三項方向與原研究假說一致，零項的模型與人類差異區間完整落在事先指定的 ±0.5 分容差內。模型估計的變化最大僅 0.028 分，312 位讀者的各組平均變化幅度為 0.279 至 1.043 分。 [事先封存的評分規格](formal-anglin-protocol.md) · [逐項分數](../results/union-observations-formal-l4/anglin/scores.json)

| 證據條件 | 首日比較 | 真人平均變化 | CogLLM 變化 | 模型減真人的 95% 區間 |
| --- | --- | ---: | ---: | ---: |
| 兩次皆反對 | T3 − T1 | −0.279 | +0.010 | [−0.137, +0.687] |
| 兩次皆支持 | T3 − T1 | +1.043 | +0.013 | [−1.418, −0.665] |
| 先反對，後支持 | T2 − T1 | −0.292 | −0.011 | [−0.025, +0.600] |
| 先反對，後支持 | T3 − T2 | +0.528 | +0.028 | [−0.805, −0.250] |
| 先支持，後反對 | T2 − T1 | +0.696 | −0.010 | [−0.985, −0.440] |
| 先支持，後反對 | T3 − T2 | −0.468 | +0.021 | [+0.262, +0.743] |

八個歷程各在 NVIDIA L4 做一次正文前向，取三個經獨立前綴切分確認的停點，共 24 處。另做 12 次截斷或替換後文的因果核對，最大相對均方根誤差為 0.0000273，低於規格的 0.001。對訓練與開發封包的 27,167 份文字檢查正規化後的連續 12 詞，2,278 個 Anglin 詞段均無匹配。第 31 輪權重由開發資料選定。八份預測在計分程式讀取逐人資料前封存。 [選定權重](../results/union-observations-settled-l4/seal.json) · [預測封存](../results/union-observations-formal-l4/anglin/forecast/seal.json) · [材料重疊查核](../results/union-observations-formal-l4/anglin/training-text-overlap.json)

本次結果適用於文字轉錄材料及固定的 Rodger 跨量尺讀出。它把政策有效性滑尺投影到 Anglin 的八點信念量尺。此次設計無法單獨估計表徵、量尺轉換及圖表呈現各自對誤差的影響。首日確定程度另有明確的問題、量尺與停點。Qiao 的資料有讀後 0–100 信心欄，公開方法與材料未提供該欄的原始題幹和錨點，也缺讀前及翌日測量，因此本次評分範圍限於首日信念。 [Anglin 原研究與量尺](https://doi.org/10.1525/collabra.133914) · [Qiao 原研究](https://escholarship.org/uc/item/1zj2p06s) · [八個歷程的封存預測](../results/union-observations-formal-l4/anglin/forecast/predictions.json)
