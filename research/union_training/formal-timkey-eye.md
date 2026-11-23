# Timkey 逐詞眼動轉移驗證

本驗證沿用 Qwen3.5-2B 與聯合訓練第 31 epoch 的凍結 `processing` 共享狀態，只以 OneStop 的英文逐詞眼動估計量測讀出。Timkey 與 Huang 使用相同句子，兩個驗證共屬一個材料家族。整批 SAP 及 Timkey 反應都不參與讀出、門檻或模型選擇。本規格固定四個眼動端點，Huang 自定步速閱讀另循其既有規格。 [模型凍結與驗證家族](holdout-plan.md) · [Timkey 等 Materials and Methods 與圖一](https://www.pnas.org/doi/10.1073/pnas.2532230123)

## 開發資料和對位

採 OneStop 封存版本 `ia_Paragraph_ordinary.csv.zip`，SHA-256 為 `8883478946ee52381e7057683c9e84dc69fcea9054acc34f0c900463a6b546e9`。使用 `practice_trial=False`、`question_preview=False`、`repeated_reading_trial=False` 的初讀，涵蓋 180 位讀者、30 篇文章、9,718 個段落試次。依實際呈現正文的 SHA-256 識別段落版本。核對 `IA_ID` 是一從首詞起連續的空白分詞索引，`IA_LABEL` 與該字串精確相同。15 個段落×難度家族有兩個實際字串，共 339 份不同正文。保留 30 篇文章，任何版本仍留在同一篇文章家族。先按三個原始閱讀批次逐批留出，以另外兩批擬合，並在各該留出批次生成讀者與文章皆未見的預測。事先固定的讀出用全部三批擬合後即對 Timkey 產生封存預測。 [OneStop 原始逐詞檔](https://osf.io/download/xkgfz/) · [資料與文字的使用條款](https://www.nature.com/articles/s41597-025-06272-2)

模型接收試次的段落原文或句子原文，一次因果 forward 擷取詞末。雲端只需要文字、匿名識別碼及 Unicode 停點。真人眼動、閱讀者、條件和觀察損失的遮罩留在本機。以固定 Qwen tokenizer 對每個 Unicode 停點核對其獨立前綴。36,983 個 OneStop 停點均通過幾何稽核。詞長只計當下詞中 Unicode 字母數。詞序採自原文起算的一起始詞序，模型特徵使用其 `log1p`，不需要知道文本總長。 [原始欄位](https://lacclab.github.io/OneStop-Eye-Movements/variables) · [固定擷取程式](../backbone_transfer/extract.py)

## 四種量測

OneStop 保留 `IA_FIRST_FIX_PROGRESSIVE=1`、`IA_SKIP=0`、`IA_REGRESSION_OUT` 為二元值、`0 < IA_FIRST_RUN_DWELL_TIME ≤ 2000` 的詞級紀錄。段落末詞缺下個文字區域，開發讀出排除該位置。

| 量測 | 僅在何種觀測計算損失 | 讀出與損失 |
| --- | --- | --- |
| 首次離開時回視 | 所有合格的首次注視 | `IA_REGRESSION_OUT`，Bernoulli 負對數似然 |
| 向前閱讀時間 | 第一次離開詞時前進 | `IA_FIRST_RUN_DWELL_TIME`，對數毫秒的高斯負對數似然 |
| 回視前注視時間 | 第一次離開詞時回視 | `IA_FIRST_RUN_DWELL_TIME`，對數毫秒的高斯負對數似然 |
| 回視路徑時間 | 第一次離開詞時回視，路徑毫秒有限且至少等於當詞注視毫秒 | `IA_REGRESSION_PATH_DURATION`，對數毫秒的高斯負對數似然 |

用詞長、詞序及十二維凍結 `processing` 狀態擬合各讀出。所有連續特徵只用擬合折均值和標準差標準化，常數維設為零。二元讀出使用 logistic 連結，時間讀出在對數尺度擬合均值，以開發殘差估計每端點共同變異數。係數沿用固定的平均損失尺度及 L2 係數 0.01，截距不受罰。兩組基線使用相同計分及校準方式，分別只包含截距或只包含詞長與詞序。真人的實際前進／回視只決定哪些條件似然可以評分，模型在同一次 forward 一併輸出機率及三個條件時間分布。 [Timkey 等眼動定義](https://www.pnas.org/doi/10.1073/pnas.2532230123) · [SR Research Data Viewer 的單詞量測對照](https://www.sr-research.com/support/printthread.php?tid=336)

## 正式比較及可用條件

Timkey 用作者封存的 368 位讀者釋出檔，並用作者處理程式的 `0 < gz ≤ 2000`、`regout` 及 `gp` 定義，失效及跳過的眼動不給時間標籤。只評前三個經典構式，以每種 24 組的歧義／控制版本作同詞對照。使用作者在 Materials and Methods 明示與 Huang 完全相同的材料。消歧詞與下一詞沿用已預先固定的配對座標。必須先核對 Timkey 自己發布的結構索引和相對位置，確定每個 `item × cond × region` 都能唯一對齊。若公開的獨立位置表不可取，用作者的釋出檔只選擇識別碼與設計位置欄位完成核對，該步驟的輸出及雜湊不得包含眼動或正誤。任何無法完整對位的端點保留為未評。 [Timkey 作者程式及資料](https://osf.io/b6rqh/) · [對照材料與 SHA-256](holdout.json)

主位置採消歧詞的向前讀時和回視前注視時間，以及下一詞的首次離開時回視和回視路徑時間。對全部 72 個同詞材料組預先產生四種預測並封存之後才套用真人觀測。每個主端點都報出按材料等權的歧義減控制差，機率以百分點，時間以毫秒，時間預測由開發對數常態的算術期望換回。模型和兩組基線也在相同留出觀測報平均負對數似然。使用種子 1729 及 2,000 次配對材料重抽樣，給出真人差、模型減真人差及模型減詞長詞序基線損失的 95% 區間，區間條件於已觀察讀者。真人方向區間包含零時方向記為未定。分構式與另個位置的數值照列於次要分析。若任何材料缺任一條件的有效支持，先報缺額，該端點的全材料主對照保留為未評。 [原研究的時間進程比較及圖三、圖五](https://www.pnas.org/doi/10.1073/pnas.2532230123)
