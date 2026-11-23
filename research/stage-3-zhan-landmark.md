# Stage 3 華語句首代詞對照

讀到句首代詞的 Qwen 表徵改善了逐筆續寫選擇的留出機率估計，但讀出高估 IC-1 的偏移、低估 IC-2，兩類偏移的排序與真人結果相反。這個差異可用來檢驗文獻效應監督能否在改善條件差的同時保留逐筆機率的準確度。 [評分結果](results/stage3-zhan-landmarks/metrics.json)

這個實驗檢查同一個因果讀出，能否在兩種真人確實看過的停點重現指涉選擇的改變。第一個停點是前句句號，第二個停點是加上的「他」或「她」。使用 Zhan 等人的實驗一，兩種提示均要求接寫解釋。原研究將自由續寫的第一個再提及對象，以及給定代詞時的指涉選擇，分別作為兩個條件的估計。 [Zhan et al.，實驗一](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0237012#sec008) · [原研究 Fig. 1](https://doi.org/10.1371/journal.pone.0237012.g001) · [Stage 2 資料](stage-2-data.md)

## 在讀出前固定的對照

正文有兩個同一性別的人物候選。二元端點是合格續寫選擇前句主語的機率。自由及代詞提示各形成一個觀測條件。每位讀者在同一材料只看一個條件，差值在材料層級比較。以 36 組材料、50 位讀者、1,651 筆已發布合格反應為分析樣本。依原研究排除所指不清、兩者、其他人及不合規的指稱形式。兩個版本永遠同折，五折驗證未見材料，同一批讀者跨折。 [Zhan et al.，實驗一 Coding](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0237012#sec008) · [資料分割](stage-2-data.md#切分與驗證範圍)

取較長版本一次因果前向的兩個末層狀態。核對分詞在介入點前一致，並以單獨截短的前綴對照數值。兩個條件共用線性讀出，正文以外沒有讀者背景、真人答案或作者動詞類型。模型只估計主語與另一候選之間的選擇，兩候選的原文區段在 Stage 2 已對齊。小規模本機前向用來核對設計；正式訓練按執行計畫使用 Colab L4。

用相同五折比較各組特徵。每折的讀出使用固定 L2 係數 1 的二項 logistic 回歸。句狀態逐列 L2 正規化。

| 條件 | 可用資訊 |
| --- | --- |
| 僅提示類型 | 是否提供句首代詞 |
| 固定句末狀態＋提示類型 | 兩個條件均使用介入前的同一表徵，再加入提示旗標 |
| 各停點狀態 | 自由提示取句末，有代詞提示取讀完代詞的位置 |
| 各停點狀態＋提示類型 | 在兩個狀態模型都提供提示旗標時，檢視改取讀完代詞的狀態所帶來的差異 |
| 作者標籤對照 | 提示旗標、作者動詞類型及兩者交互作用 |

主比較是「各停點狀態」對「固定句末狀態＋提示類型」的逐次留出 Brier 差。Brier 為機率與二元回報的平均平方差，介於 0 和 1，越低越好。第二個比較將同一材料兩個條件的主語選擇率相減，估計在 IC-1 和 IC-2 各有多大改變。另報逐材料的改變量平均絕對誤差，單位為百分點。IC-1 的動詞操弄使自由續寫較傾向主語，IC-2 較傾向另一候選。原研究報告兩類都有向主語的偏移，IC-2 的偏移較大。區間以材料家族分層配對重抽樣，固定留出預測及這一批讀者。 [Zhan et al.，實驗一 Interpretation biases](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0237012#sec008)

這個設計觀測的是在有無句首代詞的兩個續寫條件下，讀者如何選人物。模型讀出的位置精確到供給的代詞，研究量測在續寫完成後取得。若要判斷讀者在該 token 開始困惑，或在後文解開同一疑問，需要時間與事件可對齊的另外資料。

## 本機結果

| 條件 | 逐次 Brier ↓ | 逐材料偏移 MAE ↓ | IC-1 向主語偏移 | IC-2 向主語偏移 |
| --- | ---: | ---: | ---: | ---: |
| 真人觀測均值 |  |  | +8.7 pp | +33.1 pp |
| 僅提示類型 | 0.2471 | 15.6 pp | +22.8 pp | +21.1 pp |
| 固定句末狀態＋提示類型 | 0.2149 | 14.7 pp | +24.7 pp | +24.0 pp |
| 各停點狀態 | 0.2072 | 16.1 pp | +25.3 pp | +21.1 pp |
| 各停點狀態＋提示類型 | 0.2074 | 16.1 pp | +25.5 pp | +21.3 pp |
| 作者標籤對照 | 0.1606 | 9.6 pp | +10.9 pp | +31.5 pp |

主要 Brier 差為 −0.00775，固定此讀者樣本與留出預測的材料重抽樣 95% 區間為 [−0.01270, −0.00307]。兩邊均提供提示旗標時，差為 −0.00754 [−0.01236, −0.00301]。逐條件指標保留在結果檔。各停點狀態的逐材料偏移估計仍有平均 16.1 個百分點的絕對誤差。作者標籤對照有原研究的動詞分類，可用來衡量效應還有多少差距。正文讀出只用模型狀態。 [逐折留出預測](results/stage3-zhan-landmarks/predictions.jsonl) · [端點與區間](results/stage3-zhan-landmarks/metrics.json)

## 重跑

使用固定 Qwen3.5-2B 權重、本機 Apple M5 Pro、MLX 0.32.2、MLX-LM 0.31.3 與 Transformers 5.17.0。BF16 權重值轉 FP32 後，36 次主要前向共 297 token，約 2.72 秒。36 組獨立截短對照都通過，最大相對 RMS 差 0.00167，低於 0.02 容差。兩種提示的共用 token 與偏移都一致，每組恰好加入一個代詞 token。特徵快取以 SHA-256 核對。此次為本機小規模對照，未啟動 Colab。 [特徵抽取紀錄](results/stage3-zhan-landmarks/extraction.json) · [固定權重版本](https://huggingface.co/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc)

```bash
python -m research.stage3.zhan_landmark_features \
  --data .cache/cogllm/stage2/zhan \
  --model /path/to/15852e8c16360a2fea060d615a32b45270f8a8fc \
  --output .cache/cogllm/stage3/zhan-landmarks
python -m research.stage3.zhan_landmark_probe \
  --data .cache/cogllm/stage2/zhan \
  --features .cache/cogllm/stage3/zhan-landmarks/features.npz \
  --output .cache/cogllm/stage3/zhan-landmarks
```
