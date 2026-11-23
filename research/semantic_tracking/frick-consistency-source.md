# Frick 與 Schüler 的逐句矛盾判斷

Frick 與 Schüler（2026）實驗一的公開資料能把**同一讀者在一篇故事中首次按鍵回報矛盾的位置**連到當時顯示的句子。PsychoPy 每次顯示整句，讀者按 `K` 表示本句和先前內容一致，或按 `D` 表示不一致，按鍵後才顯示下一句。首句依指示應按 `K`，因此首次回報的位置從第二句起算。每次按鍵只判斷當前句與前文的關係。`D` 後出現的 `K` 評的是另一句，原先矛盾是否已解開仍未量測。讀者看見整句後才作答，句內觸發詞的位置和未受提示的自然閱讀回報仍無直接觀測。 [原文 Experiment 1，Measures、Procedure](https://doi.org/10.3758/s13421-026-01900-7)

## 來源與對齊

| 作者檔案 | 已核對的內容 |
| --- | --- |
| [`data_final_experiment_1.xlsx`](https://osf.io/download/hp9sg/) | 27,917,882 bytes。SHA-256 `439a63a228d06257661c76ed5b9b96ae3c8f57c4ac08b393844a9ef9cf517ba4`。單一工作表有 133 欄、72,723 筆資料列與表頭，其中 56,160 筆是故事呈現列。 |
| [`analysis_experiment_1_R1.Rmd`](https://osf.io/download/gbpkh/) | 第 21、29–71、75–100 行記錄試算表、原始欄位、句序、四個條件和計時轉換。作者在第 108–129 行選取目標句，再依反應時間篩選。 |
| [`Experimental_Texts_Experiment_1.pdf`](https://osf.io/download/3p2d7/) | 294,477 bytes。SHA-256 `c249217876c2c46a98a706babd3bc9644592e6bdb47abc4ca72c4fc677e13b5d`。補充材料按故事列出一致與不一致版本，精確的呈現句界可取自試算表的 `Satz` 欄。 |

逐列取 `Satz` 非空且非 `NA` 的紀錄，將 `ID` 左補零至至少六個字元，再拆成兩位故事編號、兩位條件、兩位句序及可選的 `t` 或 `s` 後綴。後綴分別標示目標句與下一句。56,160 筆的 `LoopSentences.thisN` 都等於句序減一，`StoryKey.keys` 皆為 `k` 或 `d`，`StoryKey.rt` 皆為數值。以試算表中的 `participant`、故事與條件分組，共有 180 位讀者、3,600 條完整逐句軌跡，各條件 900 條。每位讀者在四個條件各讀五個不同故事，每個故事的句序連續且無重複。每個故事版本與句序都有唯一的 `Satz` 字串。 [作者工作表](https://osf.io/download/hp9sg/) · [作者分析](https://osf.io/download/gbpkh/)

純文字條件 `01` 和 `02` 共 1,800 條故事軌跡、28,140 筆逐句判斷，`Bild` 全為 `trans.png`。20 組純文字版本各只改變一個呈現句，其句序比目標句早 7–10，目標句和其後的句子逐字相同。每位讀者只看每個故事的一種版本。這使相同目標句在兩種前文下的回報可直接比較，也保留同一讀者在同一版本中的逐句回報。 [原文 Experiment 1，Materials](https://doi.org/10.3758/s13421-026-01900-7) · [作者工作表](https://osf.io/download/hp9sg/)

## 純文字觀測

| 估計量 | 一致 `01` | 不一致 `02` |
| --- | ---: | ---: |
| 全部故事試次在目標句按 `D` | 42／900（4.7%） | 505／900（56.1%） |
| 依作者計時規則保留的目標句按 `D` | 42／890（4.7%） | 505／889（56.8%） |
| 首次 `D` 在目標句之前 | 117／900 | 120／900 |
| 首次 `D` 在目標句 | 28／900 | 458／900 |
| 首次 `D` 在目標句之後 | 20／900 | 11／900 |
| 第二句起到篇末沒有 `D` | 735／900 | 311／900 |

表中首次 `D` 排除首句，其餘按鍵完整保留。作者對目標句使用 `1000 × StoryKey.rt + 100` 毫秒，移除低於 500 毫秒的目標反應。排除的純文字試次共 21 筆，原始紀錄皆為 `K`。從第二句起計算的首次 `D` 可早於研究者安排的矛盾，未詢問這些回報指向哪個前文命題。另有 3,600 筆篇後記憶測驗按鍵。這些探測是在所有故事讀完後施測，量測對關鍵資訊的辨認。 [原文 Experiment 1，Data cleaning and inspection 與 Additional measure](https://doi.org/10.3758/s13421-026-01900-7) · [作者工作表](https://osf.io/download/hp9sg/) · [作者分析](https://osf.io/download/gbpkh/)

若後續用於模型評估，主要端點可取相同目標句在一致和不一致前文下的 `D` 機率差。每份材料只輸入截至當句的正文，兩個版本一起按故事留出，按鍵、條件與 `t`／`s` 只供評分。由於兩條件顯示的目標句相同，只看當句的模型對此配對必然給出同一結果。更早或更晚的 `D` 可用來檢查位置，但首句指示和文末仍未回報都必須依實際觀測處理。這些資料來自既有的受提示一致性判斷實驗，適用於同類敘事與任務。

論文採 CC BY 4.0。公開 OSF 專案的 `node_license` 欄為空，也無 `license` 關聯。工作表和獨立刺激檔的重用條件仍待核明，刺激中有 12 篇改編自 Cook（2014）。本輪已核對來源清冊、檔案雜湊和上述逐句紀錄。 [論文權利說明](https://link.springer.com/article/10.3758/s13421-026-01900-7#rightslink) · [OSF 專案中繼資料](https://api.osf.io/v2/nodes/pg67h/) · [資料清冊](https://api.osf.io/v2/nodes/pg67h/files/osfstorage/666ad95f77ff4c59e8e047fb/) · [刺激清冊](https://api.osf.io/v2/nodes/pg67h/files/osfstorage/666ad9726b6c8e2f2304d4d1/)

## 可重現建置

將上述工作表與 PDF 放在 `--raw` 目錄。建置保留原始呈現句字串，在句間插入兩個換行，並保存每個句末的 Unicode 排他停點。40 篇全文與 332 個去重的當句對照共 372 筆。模型輸入只有文字、由文字導出的 ID 與停點。另存的資料才包含故事配對、條件及以代碼表示的讀者軌跡。固定 tokenizer 對 956 個停點的前綴檢查全數通過。兩次資料建置逐位元相同。 [建置程式](frick_consistency_prepare.py) · [來源及輸出雜湊](../results/frick-consistency-source-audit.json)

```sh
python3 -m research.semantic_tracking.frick_consistency_prepare \
  --raw .cache/cogllm/semantic-tracking/frick-consistency/raw \
  --output .cache/cogllm/semantic-tracking/frick-consistency/prepared
```
