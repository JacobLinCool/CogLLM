# Stage 3 句級疑難的特徵對照

凍結 Qwen3.5-2B，在固定的 PLABA-EVAL 五折比較三類特徵對讀出的增益。句末狀態加長度的逐次 Brier 分數為「有無疑難」 **0.2009**、「術語不清」 **0.1987**。加入末層 surprisal 後，兩項點估計均降低約 0.0023，配對重抽樣區間均跨零。後續對照沿用句末狀態加長度。 [完整指標](results/stage3-features/metrics.json)

## 觀測與方法

採用 108 位既有讀者對 19 個文章家族中 286 句的 858 筆結構化勾選。452 筆回報至少一項疑難，406 筆明確回報無疑難，有 335 筆勾選術語不清。術語端點記錄是否勾選該項。資料的閱讀設定允許受試者查閱全文，模型輸入為正文，沒有讀者背景。訓練折和驗證折依 Stage 2 讀者連通群組劃分，文章家族與讀者均無交集。 [資料規格](stage-2-data.md) · [留出逐句預測](results/stage3-features/predictions.jsonl)

每篇文章以一次因果骨幹前向取得各位置的狀態和注意力，沒有逐 token 生成。句末基線使用 L2 正規化的 2,048 維末層狀態，加上對數轉換的字元數及空白分隔詞數。追加的三組特徵均縮減到句級。 [特徵定義](stage3/feature_signals.py)

| 特徵 | 句級取值 |
| --- | --- |
| 末層 surprisal | 用上一位置的狀態對當前 token 取負對數機率，求句內平均及最大值 |
| 第 12 層 logits | 將該層狀態經末層正規化及共用詞彙投影，計算句內平均 surprisal 和平均下一 token 分布熵 |
| 注意力摘要 | 對六個全注意力層的句末查詢取上文權重、最近八個 token 權重及依前綴長度正規化的熵，跨注意力頭及層平均 |

每個條件使用二項 logistic ridge，L2 係數固定為 1，截距免懲罰。長度和新增特徵使用訓練折的均數與標準差。在含句末狀態的條件中，長度和各組新增特徵再分別除以該組維度的平方根。主指標為逐次真人勾選與留出機率之間的平均平方誤差（Brier，越低越好）。另報告等權五折 AUROC 及同篇內有觀測高低差的句對排序，同分計半。Brier 差以 13 個有勾選資料的讀者連通群組作 10,000 次配對重抽樣，使用固定的留出預測。以下區間為描述性 95% 區間，未調整多組比較。 [評分實作](stage3/issue_probe.py) · [各折與配對區間](results/stage3-features/metrics.json)

## 結果

| 讀出特徵 | 有疑難 Brier ↓ | 術語不清 Brier ↓ |
| --- | ---: | ---: |
| 長度 | 0.2262 | 0.2213 |
| 長度＋原文／白話版標籤 | 0.2125 | 0.2008 |
| 句末狀態＋長度 | 0.2009 | 0.1987 |
| 基線＋末層 surprisal | 0.1986 | 0.1965 |
| 基線＋第 12 層 logits | 0.2028 | 0.2032 |
| 基線＋第 12 層 logits＋末層 surprisal | 0.2008 | 0.2011 |
| 基線＋注意力摘要 | 0.2016 | 0.2000 |
| 基線＋全部三組特徵 | 0.2019 | 0.2023 |

加入 surprisal 的 Brier 差為有疑難 −0.00229 [−0.00542, +0.00196]、術語不清 −0.00225 [−0.00491, +0.00116]。同篇句對排序的點估計從 72.8% 升至 74.9%，以及從 70.2% 升至 70.8%。目前證據不足以確認較低的 Brier 能在同一語料的其他分組重現。 [主比較](results/stage3-features/metrics.json)

注意力對句前文與近期 token 的權重，和均勻分布在同一前綴時的權重相關分別為 0.94、0.81。以從零開始的句首 token 索引 `s` 和句末索引 `e` 計算，均勻權重為 `s / (e + 1)` 和 `min(8, e + 1) / (e + 1)`。補充分析在兩個條件都加入句序、首句旗標、句內 token 數、可見前綴 token 數和兩個均勻權重。加上注意力後，Brier 差為有疑難 −0.00041 [−0.00535, +0.00489]、術語不清 +0.00020 [−0.00377, +0.00388]。結論涵蓋本輪測試的三個注意力摘要。 [位置控制結果](results/stage3-features/position-sensitivity.json)

本輪的標籤對齊到評分句。勾選結果支持句級疑難分析。跨句疑問首次出現的位置、指向的文字及何時解除，各需要具備相應位置或時序的既有真人觀測。

## L4 量測及重跑

使用 NVIDIA L4、固定 Qwen revision、Transformers 5.17.0、FP32 運算及 eager attention。在三種前綴長度分別做同長後文替換、截短及右補齊，九項均通過。末層及第 12 層前綴狀態的最大相對 RMS 差分別為 4.00×10⁻⁶、4.21×10⁻⁶，注意力最大絕對差為 5.01×10⁻⁶。正式抽取共 78 篇、609 句及 17,140 個非 padding token。20 個批次共用 18.61 秒（約 921 token/s），CUDA 峰值配置 7.29 GiB。以上計時涵蓋骨幹、讀取特徵和同步，排除模型載入、前綴檢查及快取寫入。 [前綴檢查](results/stage3-features/causality.json) · [環境](results/stage3-features/environment.json) · [抽取量測](results/stage3-features/extraction.json)

原雲端入口完成抽取後接著評分。精簡上傳資料缺少共用載入器所需的 `processing_ease` 欄位，雲端評分中止。在本機用下載的完整特徵配合 Stage 2 資料補算，獨立重跑的指標與逐句預測一致。L4 從配置完成至送出停止請求約 130 秒，停止後確認已釋放。用量取樣期間帳戶顯示 768.80 CU，精度為 0.01 CU，單次扣款無法由帳戶截點辨識。 [核對及雜湊](results/stage3-features/verification.json) · [L4 工作紀錄](results/stage3-features/manifest.json) · [用量](results/stage3-features/usage.jsonl)

目前入口在 L4 只抽取正文特徵，需要 `stimuli.jsonl`。將 `features.npz` 下載後，在保有完整 Stage 2 PLABA-EVAL 資料的本機執行評分。雲端工作紀錄保留當時入口版本的雜湊。

在 L4 執行

```bash
python -m research.stage3.ablation_run \
  --data /content/cogllm/data \
  --output /content/cogllm/outputs
```

以完整資料在本機評分

```bash
python -m research.stage3.issue_probe \
  --data .cache/cogllm/stage2/plaba \
  --features /path/to/downloaded/features.npz \
  --output /path/to/results
```
