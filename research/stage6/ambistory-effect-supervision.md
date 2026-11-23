# AmbiStory 的結尾差幅監督

本實驗檢驗，在義項合理性的五級分布監督中加入真人結尾差幅，能否改善同一義項在兩種後文下的預測更新幅度。沿用 AmbiStory v2 封存的 Qwen3.5-2B 特徵、形式控制、五級序位讀出與停點。同一上下文讀出在共同前綴和結尾取分布。形式控制只讀目標詞、詞義描述與共同前綴，兩個結尾下輸出相同。原測試的 155 個故事家族只提供本次研究的動機。本次將作者公開的 validation split 的 98 個家族、2,952 票作為專案留出，直至預測封存後才載入其標註。作者的 SemEval 開發階段已向參賽者公開這些標註。 [原研究 §§2.3、3.1](https://aclanthology.org/2026.semeval-1.448.pdf#page=3) · [AmbiStory 資料建置](../results/ambistory-preparation.json) · [固定 v2 讀出](../results/ambistory-fit-protocol-v2.json)

## 監督目標與選模

每個配對 \(j\) 固定一個故事家族和一個義項，按刺激識別碼排序兩個有結尾的版本。令 \(d_j\) 為第二個版本減第一個版本的真人平均合理性分數，\(\hat d_j\) 為由兩個五級預測分布的期望分數求得的同方向差值。每個配對使用同一個上下文讀出 \(W\)，輸入保留各版本在指定停點的既有因果表徵。結尾身分或原先作者要強化的義項不作為模型特徵。令 \(N\) 為訓練評分項數，\(M\) 為訓練配對數，\(\mathrm{CE}_i\) 為第 \(i\) 個評分項的原有五級分布交叉熵。固定目標為

\[
\mathcal L_\beta(W)=
\frac{\sum_{i=1}^{N}\mathrm{CE}_i(W)+\tfrac12\lVert W_{\mathrm{nonint}}\rVert_F^2}{N}
+\frac{\beta}{2M}\sum_{j=1}^{M}\left(\frac{\hat d_j(W)-d_j}{4}\right)^2.
\]

除數 4 是 1–5 級量尺上單一方向差值的最大幅度。數值擬合使用 \(N\mathcal L_\beta\)，截距沿用 v2 的處理。\(\beta=0\) 直接使用 `ambistory_fit.fit_context`。每個 \(\beta>0\) 都以同折的 \(\beta=0\) 權重為唯一初值，沿用同一最佳化方法與收斂準則，迭代上限為 3,000。

只使用作者訓練切分選 \(\beta\)。將同形詞 casefold 後以 SHA-256 排序，按雜湊次序輪流分配至五折，令同一同形詞的全部故事家族、義項及版本位於同折。每折僅用其訓練資料估計特徵縮放。固定候選為 \(\{0,1,4,16,64\}\)。只有五折都收斂、合併折外帶符號更新的 MAE 嚴格低於 \(\beta=0\)，且折外逐票 NLL 增幅不超過 0.01 nats 的正值候選才有資格。從中選折外 MAE 最小者，並以較小的 \(\beta\) 解平手。若沒有合格候選，選模判定失敗。選定後用完整作者訓練切分擬合 \(\beta=0\) 與選定的候選，封存兩套 validation 預測。 [既有訓練與讀出規格](ambistory-context-update.md)

## 留出驗收

使用 10,000 次配對重抽樣，抽樣單位為 casefold 同形詞，隨機種子為 43。每次保留該同形詞內的兩個結尾、相同義項及共用前綴。差值一律以候選減對照計算。結尾更新 MAE 對每組按固定排序的兩個版本計算 \(|\hat d_j-d_j|\)。更新幅度偏差取候選平均 \(|\hat d_j|\) 減真人平均 \(|d_j|\)。驗收須同時符合下列條件。

| 對照 | 指標 | 門檻 |
| --- | --- | --- |
| 候選結尾與同一候選的共同前綴 | 逐票 NLL 差、帶符號更新 MAE 差 | 兩項 95% 區間上界均小於 0 |
| 候選結尾與 \(\beta=0\) 結尾 | 帶符號更新 MAE 差 | 95% 區間上界小於 0 |
| 候選結尾與 \(\beta=0\) 結尾 | 逐票 NLL 差 | 95% 區間上界小於 0.01 nats |
| 候選與真人 | 平均絕對更新幅度差 | 90% 區間完全落在 ±0.5 級之內 |

逐票 NLL 沿用完整評分。共同前綴對同一義項在兩個結尾版本有相同分布，因此其結尾差值為零。這些對照可檢驗作者 validation 上的義項更新幅度及逐票預測。適用範圍是指定義項的群體合理性評分，沒有個人的疑問開始或解除時間。 [端點及各條件的共同停點](ambistory-context-update.md)

## 執行與資料分離

下列指令先封存規則，再準備資料並擬合。`AMBISTORY_EXTRACTION_DIR` 指向既有、已驗證的 v2 特徵目錄。準備階段保留無標註的 dev 和 test 輸入，擬合只取得 train 評分。正式評分在 `fit-seal.json` 完成後才接入 validation 標註。

```sh
python -m research.stage6.ambistory_effect_fit preseal \
  --output research/results/ambistory-effect-fit-protocol.json

python -m research.stage6.ambistory_effect_fit stage-inputs \
  --prepared .cache/cogllm/ambistory/causal-v3 \
  --preparation research/results/ambistory-preparation.json \
  --output .cache/cogllm/ambistory/effect-inputs

python -m research.stage6.ambistory_effect_fit fit \
  --staged .cache/cogllm/ambistory/effect-inputs \
  --extraction "$AMBISTORY_EXTRACTION_DIR" \
  --protocol research/results/ambistory-effect-fit-protocol.json \
  --output .cache/cogllm/ambistory/effect-fit
```

選出候選後，擬合輸出為 `baseline-predictions.jsonl`、`effect-predictions.jsonl`、`heads.npz`、`fit-seal.json` 和 `train-diagnostics.json`。兩套預測均沿用 AmbiStory 的 `id`、`sense`、`form_control`、`common_prefix`、`ending` 欄位，後三欄各含依 1–5 級排列的機率。若未選出候選，只封存診斷和狀態為 `no_candidate` 的收據。此時不進行留出評分。 [原預測與評分格式](ambistory_fit.py)
