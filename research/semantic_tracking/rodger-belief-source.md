# Rodger 等人的兩停點書面信念更新資料

Rodger 等人（2026）Study 1 可用來檢驗同一讀者在閱讀一段政策證據後，如何改變對**該政策是否有效**的評分。讀者先評估六項政策，再依隨機順序閱讀配給各政策的證據。每位讀者恰好遇到六種證據條件各一次，同一政策只分配到一種條件。讀者在證據完整顯示時輸入想法，接著評估證據可信度，最後再評同一項政策的有效性與支持程度。Figure 1 報各政策的初始評分，Figure 2 和補充材料 Tables B1–B2 報同一試次的評分變化及條件差。 [Study 1，Materials and measures、Procedure、Figures 1–2](https://doi.org/10.1017/bpp.2025.10028)

## 已核對的檔案與對齊

| 作者檔案 | SHA-256 | 可回查的內容 |
| --- | --- | --- |
| [`demo2_study1_data_working.csv`](https://osf.io/download/36bg8/) | `2a203401066086560d96c58509632f82c9c38d0ee6303dfbfbc2c81e0071c8ca` | 301 個唯一 `ID`，每位讀者六項政策各一組完整初始／後續評分，共 1,806 組，每種證據條件 301 組。1,806 組皆有證據頁的自由回報及後續兩個評分理由。 |
| [`demo2_study1_survey_qualtrics.qsf`](https://osf.io/download/xuqga/) | `5bb404312a59b27bd19d9ae906f6427955c77a1b3fb29a7dd8ce91630d716cd5` | 核對了六個 `baseline_<policy>` 區塊及 36 個 `evidence_<condition>_<policy>` 區塊的題目、證據 HTML、分頁和匯出欄位。每組政策與條件對應 48–52 位讀者。 |

初始有效性與支持分別取 `eff_pri_<policy>_1` 和 `sup_pri_<policy>_1`。同一列、同一政策只在一個條件有數值型後續評分，分別取 `eff_pos_<condition>_<policy>_5` 和 `sup_pos_<condition>_<policy>_1`。條件與政策把每組評分連到 QSF。證據頁的 `thought_<condition>_<policy>` 問題 HTML 包含當頁呈現的全文，分頁後的 `eval_*` 及評分區塊保留後續程序。CSV 同時保留 `thought_*`、`effJus_*`、`supJus_*`。計時欄是頁面點擊與提交時間，對齊單位為頁面。OSF 的資料與刺激元件都明列 CC BY 4.0。 [資料清冊](https://api.osf.io/v2/nodes/z4wem/) · [刺激清冊](https://api.osf.io/v2/nodes/h6ygr/) · [授權](https://api.osf.io/v2/licenses/563c1cf88c5e4a3877f9e96a/)

原論文方法寫明 1–100 分。QSF 中全部 84 個有效性與支持的滑桿實際設為 0–100 的整數，CSV 的讀前／讀後有效性有 133／168 筆零分，支持有 185／259 筆零分。建置保留零分與原量尺。 [原文 Materials and measures](https://doi.org/10.1017/bpp.2025.10028) · [核對用 QSF](https://osf.io/download/xuqga/) · [核對用 CSV](https://osf.io/download/36bg8/)

## 可測的讀者反應

主要端點可定為同一政策的讀後減讀前**有效性評分**，以原量尺分數計。用 CSV 重算，負向案例條件的有效性與支持平均變化分別為 −8.75 與 −6.87 分，和 Study 1 Results 報的 −8.8 與 −6.9 分一致。讀者的讀前分數可作為選填背景，並以同樣的讀前資訊比較不含證據文字的統計基線。政策支持可列為次要端點，亦受公平、實行難度等因素影響。 [Study 1，Figures 2、4](https://doi.org/10.1017/bpp.2025.10028) · [核對用 CSV](https://osf.io/download/36bg8/)

原文方法中的減法文字將讀前評分放在被減數位置。這裡以逐筆的「讀後減讀前」重算，結果符號和原文結果一致。兩種個案證據由作者以實際公眾意見為基礎，經 ChatGPT 生成再由團隊審核。評分來自真人，刺激涵蓋該研究的措辭與判斷程序。 [原文 Study 1，Measures 及 Quantitative results](https://www.cambridge.org/core/journals/behavioural-public-policy/article/negative-anecdotes-reduce-policy-support-evidence-from-three-experimental-studies-on-communicating-policy-ineffectiveness/9B673E6D29B154BE39D68BFBBFA5FDAA)

證據頁的自由回報與後續理由可附作定性查核。若需用來標註語意更新，要另外核對兩段回報是否提到同一個問題。研究未逐句詢問疑問或反覆測量是否已解答。讀後評分也經過一次證據可信度提問。可用停點是整頁證據後的指定判斷，句內哪個 token 讓讀者改變想法仍未量測。 [Study 1，Procedure](https://doi.org/10.1017/bpp.2025.10028)

Study 2 有 300 人，沿用 Study 1 的政策與部分證據，加入線上安全政策、修訂正向案例並增加較強的統計證據條件。Study 3 有 307 人，作者明述沿用 Study 2 的刺激，調整一個條件並新增證據診斷性的機率判斷。三項整體應視為共享材料的研究家族。線上安全政策可另查其相對 Study 1 的材料留出，僅憑研究編號無法確定所有材料未見。 [Studies 2–3，Materials and measures](https://doi.org/10.1017/bpp.2025.10028) · [三項研究材料](https://osf.io/download/vxhwj/)

## 可重現的 Study 1 建置

建置輸出六個單獨的初始政策頁及 36 個初始頁接證據頁的文字。按 QSF 的 `QuestionText` 與頁內題目順序保留文字。HTML 實體解碼，`br` 轉為換行，行內樣式移除，瀏覽器會折疊的空白合併。每頁的終點是輸出文字中的 Unicode 字元排他索引。這兩頁保留政策陳述、提問及證據頁提示。整份問卷的前導說明、其他政策、量尺控制項、後續可信度問項與讀後填答頁未放入模型文字。準備檔列出後兩頁的原始題目 ID。讀者填入的評分、理由、人口欄位及自由文字均未放入 `stimuli.jsonl`。留存在本機的 `outcomes.jsonl` 只含群集識別用代碼、對齊欄及四項評分。 [建置程式](rodger_belief_prepare.py)

`folds.json` 每折留出一項政策及其六個證據版本，測試有 301 筆既有試次。每折的訓練和測試共用這 301 位讀者。這是同研究的材料留出。兩次執行的輸出逐位元相同。固定 Qwen3.5-2B 的 Rust tokenizer 對 78 個停點逐一比對完整輸入與截短輸入，共用前綴的 token IDs 與終點均一致。 [建置程式](rodger_belief_prepare.py)

將上述兩個作者檔案存入 `--raw` 目錄。以新的 `--output` 目錄再次執行可比對輸出雜湊。

```sh
python3 -m research.semantic_tracking.rodger_belief_prepare \
  --raw .cache/cogllm/semantic-tracking/rodger-belief/raw \
  --output .cache/cogllm/semantic-tracking/rodger-belief/prepared
```
