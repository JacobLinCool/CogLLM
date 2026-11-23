# OneStopQA 基線與首輪共享讀出

**加入文章讓 OneStopQA 的真人選項預測改善。首輪共享 adapter 的增益僅出現在部分端點。** 相對獨立 adapter，共享訓練改善 PLABA 的「有疑難」勾選，其他端點的配對區間均跨零。凍結表徵的線性讀出在本輪列出的逐次誤差和句級誤差均有最低點估計，保留作參照。 [開卷指標](results/joint-first-run/onestop-probe.json) · [共享與獨立的配對結果](results/joint-first-run/joint-paired.json)

使用 Stage 2 的固定五折及 5,417 筆既有真人反應，先跑 OneStopQA 單任務對照，再比較三個任務的各自訓練與共享訓練。本輪的損失來自實際的逐次觀測。文獻彙總效應沿用原量尺評分，留待下一個對照加入訓練。 [整體訓練與驗收計畫](joint-training-plan.md)

## 固定資料與比較

- 重用 L4 上的 PLABA 句末／篇末狀態及 Zhan 自由／給定代詞停點狀態。PLABA 以兩項 1–5 分自評及兩個結構化疑難端點監督，缺少勾選的試次只計自評。Zhan 監督兩個候選指涉的選擇。
- OneStopQA 輸入使用實驗所呈現的正文、題目及顯示選項順序。完整提示 1,872 種，題目加選項 1,770 種，合計 3,642 種。所有選項文字均使用可見輸入，同一題可被多位讀者選擇。模型特徵在提示最末取得，一次因果前向，採固定 Qwen3.5-2B、FP32、無逐 token 生成。正文放在題目之前，真人端點是兩者都可見時的作答。 [實驗版本](stage-2-data.md#onestopqa-材料版本)
- OneStopQA 基線分別為訓練折的顯示位置分布、題目與選項的正規化凍結狀態、完整提示的正規化凍結狀態。兩個模型使用同一個四類線性讀出與固定 L2。正解只在得出四選一機率後，用來評估正誤。正文條件減題目條件的差用 20 組留出材料家族配對重抽樣。這個比較量測增加正文可見性後的表現，資料沒有語意疑問回報或疑問來源區段。
- 狀態讀出比較凍結線性、每個任務有各自的小型殘差 adapter，以及三個任務共用相同尺寸的殘差 adapter。任務讀出仍依量測分開。各條件使用相同的訓練分割、端點、最佳化設定及每任務更新機會。聯合訓練按觀測到的量尺分別計算損失及除以樣本權重。PLABA 的閱讀與作答需遵循原實驗自己的可見範圍，首輪只用句級回顧資料。

殘差 adapter 的 bottleneck 維度固定為 32，兩個 adapter 條件共用初始化與同一個訓練次數。用 FP32 AdamW，每折做 160 次全批更新，學習率 0.006，權重衰減 0.01。每次更新以三項任務平均損失回傳梯度。PLABA 將 1–5 分對應至 0–1，並讓自評平方誤差和有量測的勾選負對數似然等權。Zhan 使用二項負對數似然，OneStop 使用四類負對數似然。評分並未使用單一跨任務合成分數。

主要比較是共享 adapter 減每任務獨立 adapter 的各端點留出誤差。PLABA 報兩項自評 MAE 及勾選 Brier，Zhan 報逐次 Brier 與交互效應，OneStopQA 報四選一 NLL、多類 Brier 與正誤 Brier。三套資料沿用各自分組，PLABA 同時留出讀者，其餘兩套為材料留出。誤差區間用 10,000 次配對材料群組重抽樣，固定各折留出預測及讀者樣本。PLABA 的改寫差與 Zhan 的交互效應以留出預測和真人分組重算。

## OneStopQA 留出結果

1,940 筆真人作答中有 1,567 筆正確。表內 NLL 是對實際被選選項的平均負對數機率，以 nats 計。多類 Brier 是四個選項上的平方誤差之和取平均，同一題的正誤 Brier 只看正解選項的機率。均越低越好。

| 固定讀出可見資訊 | 選項 NLL ↓ | 多類 Brier ↓ | 正誤 Brier ↓ |
| --- | ---: | ---: | ---: |
| 訓練折的顯示位置先驗 | 1.3867 | 0.7502 | 0.4663 |
| 題目與實際顯示選項 | 1.3677 | 0.7406 | 0.4593 |
| 正文、題目與實際顯示選項 | 0.9844 | 0.5182 | 0.2951 |

加入正文的 NLL 差為 −0.3833，配對 95% 區間 [−0.4020, −0.3619]。這是未見來源材料的開卷反應比較，使用原研究的呈現選項。 [指標、分母與區間](results/joint-first-run/onestop-probe.json)

## 共享與獨立 adapter

| 留出端點 | 固定線性 ↓ | 獨立 adapter ↓ | 共享 adapter ↓ | 共享減獨立的配對 95% 區間 |
| --- | ---: | ---: | ---: | ---: |
| PLABA 處理容易度，句均值 MAE | 0.5824 | 0.6100 | 0.6016 | [−0.0532, +0.0305] |
| PLABA 理解自評，句均值 MAE | 0.6175 | 0.6369 | 0.6177 | [−0.0521, +0.0116] |
| PLABA 有疑難，Brier | 0.1918 | 0.2611 | 0.2320 | [−0.0413, −0.0146] |
| PLABA 術語不清，Brier | 0.1963 | 0.2200 | 0.2279 | [−0.0216, +0.0264] |
| Zhan 指涉選擇，Brier | 0.1988 | 0.1999 | 0.2011 | [−0.0075, +0.0106] |
| OneStop 選項，NLL | 0.7786 | 0.9505 | 0.9467 | [−0.0662, +0.0623] |
| OneStop 選項，多類 Brier | 0.4009 | 0.4440 | 0.4444 | [−0.0124, +0.0131] |
| OneStop 正誤，Brier | 0.2003 | 0.2106 | 0.2123 | [−0.0054, +0.0092] |

三組使用同一優化設定。上表的線性讀出用 AdamW，OneStopQA 單任務基線的讀出用固定 L2 和 LBFGS，各表的對照固定各自的優化設定。共享相對獨立 adapter 的改變依端點而異。這是固定一組共享容量、任務權重和更新次數的結果。 [逐折與逐次指標](results/joint-first-run/joint-metrics.json) · [配對比較](results/joint-first-run/joint-paired.json)

就效應而言，PLABA 的白話改寫理解自評平均高 **0.723** 分，線性、獨立、共享讀出分別估為 **0.658、0.739、0.662**。Zhan 的真人交互效應是 **+24.46 個百分點**，三組分別為 **−2.63、+5.99、+5.07**。這些是直接監督後在留出資料的診斷。新一輪把效應加入損失時，可用本輪相同資料與讀出做匹配對照。 [效應重算](results/joint-first-run/joint-paired.json)

## L4 資源與核對

34 筆正式提示試跑通過，完整抽取涵蓋 3,642 種提示及 512,429 個非 padding token。456 個批次的前向與讀取共 209.89 秒，約 2,441 token/s。CUDA 峰值配置 7.42 GiB，保留 8.90 GiB。九項前綴對照及三項正控制均通過，前綴最大相對 RMS 差為 6.62×10⁻⁶。單任務基線擬合 4.15 秒，三條件共 15 次擬合為 13.35 秒。從配置至提出停止請求約 343.72 秒，已確認釋放。帳戶顯示從 768.80 至 768.76 CU，這是整體帳戶讀數。 [實測抽取](results/joint-first-run/onestop-extraction.json) · [前綴檢查](results/joint-first-run/onestop-causality.json) · [執行與釋放](results/joint-first-run/manifest.json) · [用量](results/joint-first-run/usage.jsonl)

在本機從 15 份權重重建五折預測，最大機率或分數差為 6.20×10⁻⁶。用個別反應重新計算主要分數後一致，配對比較重跑逐位元組一致。CPU 合成試跑中，將所在折留出標籤改值後，預測不變。改動 OneStop 訓練反應時，兩個其他任務的獨立 adapter 預測不變，共享 adapter 的預測會改變。 [重播與稽核](results/joint-first-run/verification.json) · [逐次分數複算](results/joint-first-run/independent-metric-verification.json)

## 並行準備的詞級端點

CompLex 固定發布的 10,800 筆英語詞與多詞片語評分中，10,740 筆可唯一對到原文字元偏移。56 筆在考慮大小寫後仍有多個候選，3 筆無符合邊界的候選，1 筆只有忽略大小寫後匹配。已核對 1,101 筆官方測試輸入與評分檔。官方 train 和 test 跨兩任務重複 223 個逐字相同句子，CogLLM 的詞級留出需按句子分組。 [字元定位與版本雜湊](results/joint-first-run/complex-audit.json) · [重跑程式](joint/complex_audit.py)

原量測請真人在整句可見時評指定目標的詞彙難度，將五點評分映射至 0–1 再彙總。未評詞屬未觀測。固定版聲明 CC BY 4.0，稽核另列上游來源及尚未逐篇核對的生醫文章。這批資料可作後續局部詞彙難度端點，首次困惑停點及同一疑問的解除仍需相應時序觀測。本輪已完成來源和定位稽核。 [原始方法](https://aclanthology.org/2021.semeval-1.1.pdf#page=3) · [固定發布](https://github.com/MMU-TDMLab/CompLex/tree/4b31c2bd34da4c1265f9db3d727d95ee23099f14)

## 重跑入口

在本機以 `python -m research.joint.prepare --data .cache/cogllm/stage2 --output .cache/cogllm/joint/data` 重建觀測計數與無標籤提示。L4 先執行 `research.joint.onestop_features`，再用同一特徵執行 `research.joint.onestop_probe` 及 `research.joint.train`。訓練另取 Stage 3 及 Stage 4 先前的 PLABA、Zhan L4 狀態。下載後用 `research.joint.compare` 重算逐端點及文獻效應對照。 [資料建置](joint/prepare.py) · [抽取](joint/onestop_features.py) · [開卷基線](joint/onestop_probe.py) · [共享訓練](joint/train.py) · [留出對照](joint/compare.py)
