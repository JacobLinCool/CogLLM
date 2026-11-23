# ExpMRC 到達機率的回溯校準結果

在 165 篇文章、555 道指定問題及 1,110 份等權人工參考上，校準後的逐 token 讀出得到 **3.0760 nats 區間 NLL、0.1836 累計 Brier**。以文章分為五折，每折 33 篇，只用另外四折擬合兩個全域參數。沿用原分析的資料排除、停點與終點。兩項指標愈低愈好。 [分析規格](expmrc-calibration.md) · [彙總結果](../results/expmrc-calibration.json)

| 讀出 | 區間 NLL（nats） | 累計 Brier |
| --- | ---: | ---: |
| 校準後逐 token | 3.0760 | 0.1836 |
| 原逐 token | 5.0903 | 0.4497 |
| 原題幹 | 3.1963 | 0.1942 |
| 校準後減原逐 token | −2.0143 [−2.1246, −1.8983] | −0.2661 [−0.2872, −0.2444] |
| 校準後減原題幹 | −0.1203 [−0.1753, −0.0655] | −0.0107 [−0.0188, −0.0032] |

括號內為固定折外預測後，以文章重抽樣 10,000 次所得的 95% 描述區間。第一停點有 4.23% 的人工參考已完整覆蓋，平均預測機率為校準後逐 token 5.16%、原逐 token 0.086%、原題幹 3.19%。單調轉換在 698 組原規格同文同停點配對保留每組順位，維持 435 勝、263 負、0 平。555 條校準軌跡的各區間到達機率與末端剩餘機率合計均為 1。 [原評分結果](../results/expmrc-transfer-l4/scores.json) · [彙總結果](../results/expmrc-calibration.json)

前次轉移實驗已評讀同一批 ExpMRC 標註，本次結果是回溯折外估計。評分對象是給定題目後，讀到何處已包含標註證據。新文章上的絕對機率表現仍需獨立驗收。

本機僅執行單 CPU 執行緒的兩參數擬合與評分，最高實測程序 RSS 為 **82.7 MiB**。兩次執行的彙總 JSON 逐位元相同，程式計時分別為 0.41 和 0.42 秒。庫根目錄的重跑指令沿用既有依賴快取，版本為 Python 3.14.7、NumPy 2.5.5、SciPy 1.17.1。 [執行程式](expmrc_calibrate.py)

```sh
PYTHONPATH=.cache/cogllm/anglin/deps python3 research/semantic_tracking/expmrc_calibrate.py \
  --assay-root .cache/cogllm/exp-mrc/v1 \
  --forecast-dir research/results/expmrc-transfer-l4 \
  --original-protocol research/semantic_tracking/expmrc-transfer.md \
  --calibration-protocol research/semantic_tracking/expmrc-calibration.md \
  --heads research/results/qasper-factorial-run/heads.npz \
  --output /tmp/expmrc-calibration-replay.json
```
