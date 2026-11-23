# CLEAR 英文篇章閱讀容易度留出結果

在 410 篇未參與校準的英文材料上，將固定 Gradio 候選的句子容易度加入表面特徵，使篇章排序與教師容易度估計的 Spearman 相關從 **0.608 提升至 0.691**。差值為 **+0.083**，按來源配對重抽的 95% 區間為 **[+0.051, +0.118]**，通過事先固定的門檻。直接使用現行 App 句級分數做詞數加權平均，無須在 CLEAR 擬合，相關為 **0.661**，相對表面讀出多 +0.053 [+0.012, +0.096]。 [固定規格](clear-reading-ease-protocol.md) · [封存指標](../results/clear-reading-ease-l4/metrics.json)

| 留出材料 | 篇數 | 直接用 App 分數 | 表面讀出 | 表面加 App | 表面加 App 的相關改善及 95% 區間 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全部 | 410 | 0.661 | 0.608 | 0.691 | +0.083 [+0.051, +0.118] |
| 說明文 | 291 | 0.635 | 0.537 | 0.690 | +0.153 [+0.092, +0.218] |
| 文學 | 119 | 0.101 | 0.067 | 0.133 | +0.066 [−0.048, +0.180] |

兩個校準讀出使用相同 600 篇材料，同一來源元件無跨分割。表面特徵是正文空白詞數的對數、每句平均詞數、英文字詞平均字母數和長詞比例。相對表面讀出的留出 RMSE 從 0.821 降至 0.734，單位是 CLEAR 發布的 Bradley–Terry 分數。排除校準組唯一一篇與既有材料共用 12–29 詞的摘錄、重新擬合後，相關增益仍為 +0.083 [+0.051, +0.118]。點估計、兩個 RMSE 和主要對照的 5,000 次重抽區間均獨立重算。兩次完整評分產物逐位元相同。 [資料及分拆](clear-reading-ease-source.md) · [校準資料分母與權重](../results/clear-reading-ease-l4/calibration.json) · [獨立核對](../results/clear-reading-ease-l4/independent-check.json)

這個端點是教師看完整篇摘錄後，對國小三年級至高中讀者可能遇到的難度所做的兩兩比較。發布分數可驗證篇章順位。英文說明文有明確增益，文學的增益區間跨零。該資料沒有實際學生的逐句理解或難度回報，因而無法校準 App 的單句顏色或定位疑問起始與解除。這次校準讀出只供驗證使用。 [CLEAR 原研究 Study 1](https://link.springer.com/article/10.3758/s13428-022-01802-x)

## 同一候選的前向與介面驗收

同一 Qwen3.5-2B 候選在一台 Colab L4 上分析 1,010 篇。每篇內容一次 FP32 因果前向，12,743 個句末的原文座標與 Qwen token 索引全部與事前封存結果一致，分數沿用正式 App 的句末權重。三份額外前綴對照截去未讀後文，已讀句分數的最大差為 0.00000015。模型收到正文及定位資料，教師估計在取回預測後接入本機評分。 [擷取收據](../results/clear-reading-ease-l4/extraction.json) · [逐句封存](../results/clear-reading-ease-l4/forecasts.jsonl) · [前綴核對](../results/clear-reading-ease-l4/causality.json)

使用同一模型程序在 Gradio 實際提交兩篇測試文章，每篇一次前向。替換文章後清除先前的上文指向查詢，並顯示兩個句子底色和 18 個詞彙底線。八次後續選取與檢視操作共做零次前向。已核對原文 Unicode、標記位置及新版的「模型估計」與「篇內相對色階」說明。 [實際全文標色](../results/clear-reading-ease-l4/preview.html) · [HTTP 驗收](../results/clear-reading-ease-l4/ui-verification.json)

正文前向及讀出共 114.01 秒，整個實驗階段 160.67 秒。峰值 GPU 保留記憶體約 7.09 GiB。自 L4 配置到釋放確認共 219.01 秒。按配置期間和帳戶顯示的 1.94 CU／小時估算約 0.118 CU，實際帳戶扣點無法由本次畫面精度及入帳時間核定。 [執行、用量與釋放收據](../results/clear-reading-ease-l4/run-verification.json)

## 重跑

由 [固定工作簿與準備程式](clear-reading-ease-source.md#正式材料準備) 重建三個資料檔，核對它們與原準備收據的雜湊。遠端由 `cuda_verification_batch prepare-clear-ease` 綁定已驗證的 Gradio 候選與 CLEAR 停點，再執行該工作目錄。以下從專案目錄以已取回的 L4 封包重算評分，無需載入語言模型。 [Colab 執行入口](../gradio_app/cuda_verification_batch.py) · [評分入口](clear_reading_ease_score.py)

```sh
clear_root=.cache/cogllm/clear-reading-ease
clear_l4="$clear_root/l4/cogllm-cuda-checks-20261121T074802Z-8726bc98/outputs/clear-ease"
python3 -m research.stage6.clear_reading_ease_score \
  --prepared "$clear_root/prepared-v1" \
  --forecasts "$clear_l4/forecasts.jsonl" \
  --seal "$clear_l4/seal.json" \
  --output "$clear_root/scoring-reproduction"
```
