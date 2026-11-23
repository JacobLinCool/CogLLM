# Stage 4 華語句首代詞的效應監督

**弱監督縮小了留出材料的交互效應誤差 5.21 個百分點，但只重現了很小的差距。** 真人兩類材料的偏移相差 +24.46 個百分點，讀出從 −4.09 變為 +1.12。逐次 Brier 差為 +0.00104，材料重抽樣區間跨零。 [完整結果](results/stage4-zhan-weak/metrics.json)

本輪檢驗弱監督能否讓讀出重現兩類華語材料在句首代詞出現後的不同指涉偏移。固定 Stage 3 的 Qwen3.5-2B 權重和五折。在每篇材料一次因果前向中，於前句句號及下一句代詞取最後一層狀態，分別讀出合格續寫選擇前句主語的機率。沿用 36 組材料和 1,651 筆既有試次，同一批 50 位讀者跨折。 [Stage 3 對照](stage-3-zhan-landmark.md) · [原研究 Fig. 1](https://doi.org/10.1371/journal.pone.0237012.g001)

## 結果前固定的比較

每折的每個訓練材料和提示條件各取得 4 筆直接標註。用種子 0–11 產生 12 個固定標籤抽樣。少量直接和弱監督兩組共用相同的逐次標籤。弱監督額外使用該折其餘訓練反應算成**一個彙總效應**。完整直接監督使用該折全部合格反應。三組使用同一種輸入、同一個共享 logistic 讀出和相同正規化強度。留出材料的反應只供最終評分。

兩個條件在每個材料內的差是加上代詞後的主語再提及率，減去自由提示的主語再提及率。各動詞類的差以材料等權平均。單一主要效應 $I$ 是 **IC-2 的偏移減去 IC-1 的偏移**。模型機率 $\hat p$ 依相同規則算成 $\hat I$。動詞類型用於彙整這個差。原研究同時有階層模型及依讀者彙整的結果，本輪的目標採依材料等權的可重算效應。 [原研究 Experiment 1，Interpretation biases](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0237012#sec008) · [效應損失規格](mathematical-formulation.md#從文獻效應回傳梯度)

對每一折，讀出將各位置的末層向量 L2 正規化，並最小化

$$
\mathcal L
= \frac{\mathrm{NLL}_{\mathrm{direct}}}{N_{\mathrm{direct}}}
+ \frac{\lVert w\rVert_2^2}{2 N_{\mathrm{full,train}}}
+ \frac{\lambda}{2}(\hat I-I)^2 .
$$

截距免懲罰。弱監督固定 $\lambda=1$，兩組直接監督 $\lambda=0$。同一折分母 $N_{\mathrm{full,train}}$ 固定，三組的平均似然正規化可比較。以 Colab L4 快取官方 FP32 表徵，再用雙精度訓練讀出及查核收斂。Qwen 全程凍結。

逐次 Brier 與主要交互效應的絕對誤差為兩個共同判準，另外報各條件 Brier、兩類偏移、每組材料偏移的平均絕對誤差。先算每個抽樣的得分，再對 12 個抽樣取平均。材料重抽樣保留兩種提示、動詞分層、同一組留出預測及讀者樣本。主要比較是加弱監督是否縮小交互效應誤差，以及逐次機率誤差有何變動。這是同一研究的留出材料對照。

## 留出結果

| 監督條件 | 逐次 Brier ↓ | IC-1 偏移 | IC-2 偏移 | IC-2 − IC-1 | 交互效應絕對誤差 ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 真人觀測 |  | +8.68 pp | +33.14 pp | +24.46 pp |  |
| 少量逐次 | 0.21085 | +24.94 pp | +20.85 pp | −4.09 pp | 28.55 pp |
| 少量逐次＋彙總效應 | 0.21189 | +22.63 pp | +23.75 pp | +1.12 pp | 23.34 pp |
| 完整逐次 | 0.20714 | +25.25 pp | +21.07 pp | −4.18 pp | 28.64 pp |

每折少量監督使用 224–232 筆，完整條件使用 1,288–1,329 筆。對應同一組少量標籤，弱監督的效應絕對誤差差值為 −5.21 個百分點（配對材料 95% 區間 [−6.38, −4.02] 個百分點）。12 個抽樣中，弱監督有 8 個估出正向交互效應，少量逐次有 1 個。弱監督的逐次 Brier 增加 0.00104（[−0.00059, +0.00277]），逐材料偏移 MAE 從 17.14 降至 15.75 個百分點。完整逐次標籤在三組中有最低逐次 Brier，平均條件效應仍偏離真人。 [抽樣結果與重抽樣區間](results/stage4-zhan-weak/metrics.json) · [逐材料留出機率](results/stage4-zhan-weak/predictions.jsonl)

弱監督的折內效應目標平均為 +24.48 個百分點，擬合後的折內效應平均為 +16.21，合併留出材料後僅剩 +1.12。本輪支持條件差可透過彙總效應往真人方向移動，且在未見材料上只保留小幅度的改善。下一個研究家族需保留獨立以檢查遷移。這批既有資料衡量的是兩種提示下的指涉選擇，疑問發生和解除需要對應觀測。 [訓練診斷](results/stage4-zhan-weak/fitting.json)

## 資源與核對

Colab NVIDIA L4 使用 PyTorch 2.14.1、Transformers 5.17.0、FP32 骨幹及 SDPA。10 個主要批次處理 36 組共 297 token，前向與取出狀態約 5.15 秒，峰值 CUDA 配置約 7.06 GiB。36 組截短對照均通過，最大相對 RMS 差 4.35×10⁻⁶。125 次 FP64 讀出擬合約 36.13 秒，峰值 CUDA 配置約 20.2 MiB。配置至送出停止請求約 114.7 秒，之後已確認釋放。活動時帳戶顯示 1.94 CU／小時，各次用量截點為 768.80 CU，無法從這些截點確定單次扣款。 [資源紀錄](results/stage4-zhan-weak/manifest.json) · [抽取結果](results/stage4-zhan-weak/extraction.json) · [用量](results/stage4-zhan-weak/usage.jsonl)

將每一折的留出答案全部反轉，該折的逐次標籤計數及效應目標維持不變。下載後用 NumPy 從所有 125 組權重重算梯度、效應及留出機率，機率最大差 2.22×10⁻¹⁶。兩份評分輸出重跑後逐位元組一致。 [驗證與雜湊](results/stage4-zhan-weak/verification.json)

## 重跑

在本機用 [Stage 2 資料](stage-2-data.md) 重建折內逐次標籤計數及剩餘反應的彙總效應。本機保有完整的逐次作答。將 `design.json` 與 `stimuli.jsonl` 傳至 L4 的資料目錄。

```bash
python -m research.stage4.zhan_weak prepare \
  --data .cache/cogllm/stage2/zhan \
  --output .cache/cogllm/stage4/design.json
```

在 L4 先抽取狀態，再訓練讀出。

```bash
python -m research.stage4.zhan_features \
  --data /content/cogllm/data \
  --output /content/cogllm/outputs
python -m research.stage4.zhan_weak fit \
  --design /content/cogllm/data/design.json \
  --features /content/cogllm/outputs/features.npz \
  --output /content/cogllm/outputs --device cuda
```

下載 `heads.npz`，用完整真人反應評分。

```bash
python -m research.stage4.zhan_weak_score \
  --data .cache/cogllm/stage2/zhan \
  --heads /path/to/downloaded/heads.npz \
  --output /path/to/results
```
