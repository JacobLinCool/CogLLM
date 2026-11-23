# QASPER 有證據與不可回答的聯合校準結果

把可定位證據和明示不可回答的標註一起納入時間校準後，逐 token 讀出在新 QASPER 論文上的區間 NLL 從原始的 **10.153 降到 4.370 nats**，差值的 95% 文章配對區間為 [−7.856, −3.909]。額外擬合固定的尾端剩餘機率參數後，NLL 為 4.385，差 +0.0149 [−0.0210, +0.0580]，未通過預先設定的改善門檻。NLL 越低代表分配給人工標註的機率越高。 [固定分析規格](qasper-evidence-tail.md) · [預先設定的比較](../results/qasper-evidence-tail-l4/scores.json) · [補充配對比較](../results/qasper-evidence-tail-l4/secondary-verification.json)

| 逐 token 讀出 | 留出 NLL | 有證據的 NLL | 明示不可回答的 NLL | 累積 Brier |
| --- | ---: | ---: | ---: | ---: |
| 原讀出 | 10.153 | 9.528 | 18.104 | 0.3497 |
| 原 ExpMRC 校準 | 38.407 | 36.014 | 67.301 | 0.5766 |
| QASPER 混合標註，校準時間 | 4.370 | 4.674 | 1.340 | 0.1554 |
| 同上，加上尾端剩餘機率參數 | 4.385 | 4.693 | 1.306 | 0.1551 |

新校準的逐 token 讀出，相對同樣校準的位置基線，NLL 差 −0.250 [−0.364, −0.138]。相對段落交互讀出差 −0.002 [−0.020, +0.018]。因此這批資料支持正文資訊改善位置估計，逐 token 交互在已知問題上的增益仍未優於段落交互。各組單調轉換均保留嚴格同文配對的累積強度順位。 [逐組評分](../results/qasper-evidence-tail-l4/scores.json) · [同校準對照](../results/qasper-evidence-tail-l4/secondary-verification.json)

## 同語料的有證據對照

主分析完成後，補做一個同語料控制。兩組都在同一批 QASPER 開發材料上用相同的兩參數模型，控制組只用可定位證據。納入不可回答標註後，留出中這類標註的 NLL 從 2.570 降到 1.340，配對差 −1.230 [−1.781, −0.667]。全部標註的 NLL 為 4.370 對 4.366，配對差 +0.004 [−0.129, +0.121]。因此新語料校準後的整體改善，還無法歸因於加入不可回答標註。這些標註的額外益處集中在模型給不可回答情形分配的機率。 [補充實驗與區間](../results/qasper-evidence-positive-control.json)

## 留出範圍與重現

官方 train 的 28 篇、76 題內有 54 份有效人工標註，其中 44 份有可定位證據，10 份明確不可回答。官方 test 的 25 篇、78 題內有 73 題、165 份有效標註，其中 149 份有證據，16 份明確不可回答。圖表、缺少文字證據及無法唯一定位的標註另計。兩組文章與原 QASPER 驗證用過的 110 篇論文沒有識別碼或正規化標題重合。所有論文保留完整發布的文字。 [留出與排除原因](../results/qasper-evidence-tail-l4/assay.json) · [原資料與標註方法](https://aclanthology.org/2021.naacl-main.365/)

四組共用凍結的 Qwen3.5-2B 骨幹及原讀出權重，L4 對 53 篇正文各做一次模型前向。文章及題幹共 136,234 tokens，全部 4,836 個計分停點和前綴對齊通過。前向與擷取為 58.8 秒。原始預測封存後，在本機用 train 擬合並存檔，然後接上 test。QASPER 的配適和評分重跑逐位元一致，另用直接機率公式重算全部 16 個 NLL。L4 已釋放。 [L4 前向時數](../results/qasper-evidence-tail-l4/extraction.json) · [配適權重](../results/qasper-evidence-tail-l4/fit.json) · [釋放與重跑收據](../results/qasper-evidence-tail-l4/run-verification.json)

## 回溯套用到 Natural Questions

將封存的 QASPER 時間校準套到已評讀過的 Natural Questions，逐 token 區間 NLL 為 3.153，低於該批原讀出的 7.776 和原 ExpMRC 校準的 27.210。加上尾端剩餘機率參數後為 3.263，差 +0.109 [0.048, 0.188]。本對照未使用 Natural Questions 標註來擬合，但其先前結果已用於設計這輪實驗。兩個語料的端點均是指定問題的證據可見時間。讀者的自發疑問及何時意識到答案，仍待有同一人多停點的直接驗證。 [回溯轉移結果](../results/qasper-evidence-tail-l4/nq-replay.json) · [原 Natural Questions 結果](nq-evidence-transfer-results.md)

在同一回溯驗證中，只用 QASPER 的有證據標註擬合時間，得到 3.565 nats。混合標註相對這組同語料對照的差為 −0.411 [−0.600, −0.223]，其中空標註的差為 −0.817 [−1.211, −0.397]。 [補充回溯比較](../results/qasper-evidence-positive-control-nq-replay.json)
