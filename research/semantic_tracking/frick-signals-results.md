# 逐 token 詞表與注意力訊號的句末矛盾回報驗證

Frick 與 Schüler 的 20 組德文故事中，在相同目標句首次按鍵回報矛盾的機率，不一致前文比一致前文高 **48.27 個百分點**。原七特徵讀出估計 **18.95**，加入詞表與注意力訊號後估計 **16.36**。主組未通過目標句幅度改善門檻。 [固定規格](frick-signals-protocol.md) · [真人觀測與材料對齊](frick-consistency-source.md) · [封存結果](../results/frick-signals-l4/results.json)

## 留出結果

每折一起留出同一故事的兩個版本。1,800 條真人軌跡都保留，首次按鍵位置從第二句起算，篇末仍無回報列為未觀察到事件。條件差先在各故事內估計，再將 20 組故事等權平均。表中的 NLL 越低，表示模型給真人首次回報實際位置的機率越高。區間以故事及讀者交叉重抽 10,000 次。 [封存預測](../results/frick-signals-l4/forecasts.jsonl) · [擬合設定](../results/frick-signals-l4/fits.json)

| 讀出 | 目標句首次回報差（百分點） | 相對原讀出的幅度絕對誤差改善及 95% 區間（百分點） | 整篇 NLL（nats／篇） | 相對原讀出的 NLL 差及 95% 區間 |
| --- | ---: | ---: | ---: | ---: |
| 原七特徵 | +18.95 | — | 1.5939 | — |
| 詞表與注意力 | +16.36 | −2.59 [−5.36, +0.03] | 1.5497 | −0.0441 [−0.0872, +0.0020] |
| 只加詞表 | +15.25 | −3.70 [−6.70, −1.08] | 1.5902 | −0.0037 [−0.0416, +0.0362] |
| 只加注意力 | +20.01 | +1.06 [−0.05, +2.20] | 1.5640 | −0.0299 [−0.0526, −0.0059] |

主組通過相對原讀出的整篇 NLL 非劣門檻。兩條件的提前回報誤差區間上界分別為 +10.94、+11.56 個百分點，超出預定的 ±10 範圍。只加注意力的事先指定拆解中，整篇 NLL 低於原讀出，兩條件的提前回報也通過等效門檻。目標句的幅度改善區間跨零，預測仍低估真人差幅約 28.3 個百分點。這批資料已用於多輪 Frick 模型比較，注意力增益仍需要新材料驗收。 [門檻及各條件分布](../results/frick-signals-l4/results.json)

補充診斷限定在目標句前仍未回報的讀者。真人與四個讀出在 20 組故事都呈現同一方向，不一致版的回報機率較高。真人的不一致版條件回報率平均是 59.14%，原七特徵為 30.15%，兩種訊號合用為 27.74%，只加注意力為 30.73%。即使將不一致版的提前回報全數去除、把一致版的目標句首次回報降至零，固定注意力讀出的目標句機率仍只能得到 30.73 個百分點，低於真人全體的 48.27。僅校準更早位置無法修正這個差距。各模型的條件機率直接由封存的目標句 logit 取 sigmoid，真人分母為在各故事版本到達目標句且尚未回報的人數。 [補充診斷方法](frick-target-hazard-diagnostic.md#目標句分解) · [本次封存預測](../results/frick-signals-l4/forecasts.jsonl)

## 前向、座標與資源

使用 Gradio 候選的同一凍結 Qwen3.5-2B。40 篇正文各做一次因果前向，取得 9,386 個 token 的四項訊號，以及 624 個句末狀態。對照證實擷取訊號時的隱藏狀態與原前向逐值一致。替換未讀後文時，保留前綴的四項訊號也逐值一致。截去未讀後文時，四項訊號的最大絕對差不超過 0.000013。獨立分詞重建全部原文座標，並核對 20 組目標句的 token 相同。各組擬合使用相同的舊句末狀態，重新擷取的狀態用於核對。原讀出逐位元重現，刪除留出材料的目標及回報數後，新增讀出的預測也完全相同。 [前向檢查](../results/frick-signals-l4/causality.json) · [獨立座標檢查](../results/frick-signals-l4/independent-prefix-check.json) · [讀出核對](../results/frick-signals-l4/axis-audit.json)

L4 正文前向與讀出共 10.06 秒，完整擷取 36.07 秒，峰值保留顯示記憶體 7.13 GiB。從配置到確認釋放為 113.73 秒。兩次本機 CPU 擬合與評分各約 6.5 秒，峰值 RSS 約 235 MiB，封存產物逐位元相同。另從封存 logit 及逐次回報用標準函式庫重算各組幅度和整篇 NLL。 [擷取規格與雜湊](../results/frick-signals-l4/extraction.json) · [獨立評分及釋放收據](../results/frick-signals-l4/run-verification.json)

每個 token 有可對回原文的模型訊號。真人作答點都在整句已顯示之後，資料可評估哪一句引發首次回報及其機率，無法辨識句內首次察覺的詞，也未詢問該回報指向哪個舊句或何時解疑。這組讀出保留為研究結果，Gradio 的句級及詞彙顯示仍使用其原有讀出。 [原研究的流程與測量](https://doi.org/10.3758/s13421-026-01900-7)

## 重現

依 [來源建置方式](frick-consistency-source.md#可重現建置) 準備真人檔案。擷取使用 `cuda_verification_batch.py` 的 `prepare-frick-signals`，以原擷取封包的 `qwen-packet/stimuli.jsonl` 和已驗證的 Gradio L4 工作目錄建立模型輸入。評分可直接重用 L4 封存。以下從研究專案目錄操作，需安裝 NumPy 與封存版本的 Hugging Face Tokenizers。 [L4 執行入口](../gradio_app/cuda_verification_batch.py) · [訊號擷取](frick_signals_extract.py) · [巢狀評分](frick_signals_score.py)

```sh
frick_cache=.cache/cogllm/semantic-tracking/frick-consistency
frick_old="$frick_cache/l4/cogllm-cuda-checks-20261121T041742Z-f6cc9d6e/outputs/projection"
frick_new="$frick_cache/signals-l4/cogllm-cuda-checks-20261121T072128Z-845a9a12/outputs/frick-signals"
frick_tokenizer="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc/tokenizer.json"
python3 -m research.semantic_tracking.frick_signals_score \
  --assay "$frick_cache/prepared" \
  --archived-extraction "$frick_old" \
  --archived-packet "$frick_cache/qwen-packet" \
  --signals "$frick_new" \
  --signal-stimuli "$frick_cache/signals-input/stimuli.jsonl" \
  --signal-assay research/results/frick-consistency-l4/extractor-preflight.json \
  --signal-preflight research/results/frick-signals-l4/extractor-preflight.json \
  --tokenizer "$frick_tokenizer" \
  --output "$frick_cache/signals-reproduction"
```
