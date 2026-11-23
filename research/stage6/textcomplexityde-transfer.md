# PLABA 句末讀出對德語自評的外部排序驗證

這項測試檢查凍結的 PLABA 句末讀出能否在德語單句上保留兩個端點的排序。TextComplexityDE 的德語學習者看完整的單句，分別評句子複雜度與自己理解得如何。前者量尺是 1＝很容易、7＝很複雜。後者是 1＝完全理解、7＝完全沒理解。發布檔提供每句各端點的有效票數與平均分。〔研究假說〕句末狀態能改善兩種真人自評的排序，相對只從句子長度估計的對照仍有增益。 [原研究 §§II.B–C](https://arxiv.org/pdf/1904.07733) · [作者發布及欄位](https://github.com/babaknaderi/TextComplexityDE#repo-details)

## 封存材料與模型

使用作者發布的 `source.csv`、`ratings.csv` 和 `LICENSE` 原始位元組。1,000 筆評分列均能接到來源句 ID。排除 100 筆標為 `Leichte Sprache`、缺少 Wikipedia 來源連結的控制句，以及 125 筆兩個發布檔的句子字串有差異的 Wikipedia 列。保留 775 個兩檔全文字串一致的原句，分屬 23 篇德語 Wikipedia 文章。以 Unicode NFKC 正規化、case folding 並保留詞與數字來篩除同文。保留句子互不重複，與 PLABA 609 個發布句子也沒有相同的正規化詞序。源文章的歸屬、文章題名和連結由各列的作者欄位驗證。這組材料不事後依真人評分或模型表現挑選。 [準備程式](textcomplexityde_transfer.py) · [作者原始材料](https://github.com/babaknaderi/TextComplexityDE/tree/master/TextComplexityDE19)

作者資料庫附 MIT 授權。Wikipedia 句子保留原文及原始文章連結，依來源文字的 CC BY-SA 條件署名與使用。模型輸入包含純正文、文章署名與先前固定的權重。真人評分與票數獨立留在本機。 [作者 MIT 授權](https://github.com/babaknaderi/TextComplexityDE/blob/master/LICENSE) · [Wikimedia 使用條款 §7](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content) · [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)

使用現有 PLABA `qwen_plus_length` 序位讀出，凍結原始 `weights` 陣列。對照使用相同 PLABA 開發資料、相同五分類似然與 L2 罰項擬合的長度讀出。兩者的字元及詞數尺度沿用 PLABA。每個德語句子單獨輸入 Qwen3.5-2B，做一次 FP32 因果前向，讀取精確對齊的最後 token 狀態。以兩種讀出的 1–5 期望分數排序。這個封存測試固定兩種讀出，不以外部評分調整模型、校準或候選集。 [PLABA 序位權重驗證](../results/causal-plaba-ordinal/manifest.json) · [序位訓練與長度對照](../stage3/causal_ordinal.py)

## 預先定義的評分

按來源文章分組。每篇文章內只比較真人平均分不同的句子對，按數值方向將模型分數轉成難度，模型同分計半。一致率的範圍是 0–1，越高越好。各有合格比較對的文章等權平均。至少 95% 的預定句子須有精確停點與有效封存預測。用種子 43 依文章做 10,000 次配對重抽樣。兩個端點各自的「句末讀出減句長讀出」一致率差的百分位 95% 區間下界都大於零，才通過這項外部排序測試。

本測試使用德語學習者、Wikipedia 單句及群體平均自評。通過後的結論限於這兩種排序在該材料上相對句長的增益。PLABA 的英文篇章情境、1–5 機率分布與改寫效應幅度仍各自需要相應的人類觀測。這裡不以不同尺度互相校準，也不將非母語學習者的平均分作個人預測真值。

## 其他資料的可用範圍

MedReadMe 讓標註者先排序六個句子，為每句評閱讀所需的 CEFR 程度，也可打開鄰句。研究的 1,924 個開發與測試分數為兩位標註者的平均。實際作者 CSV 有 4,504 列，包括 2,580 訓練、784 開發及 1,140 測試，比論文描述少 16 筆訓練列。已依作者 CSV 對 PLABA 的 609 句核對正規化全文、去標點詞序、PLABA 全文子字串及 5 詞片段覆蓋，未發現相同句子或 25% 以上的 5 詞片段覆蓋。CSV 缺個人票、排序組、篇章或原文配對鍵，也缺點開上下文的紀錄，因此原始研究所述的文章家族與對單句的實際曝光無法由發布檔確認。作者要求引用並提供資料下載。跨來源的文本再發布條件需逐一核實。這份資料可研究 CEFR 可讀性的次要轉移，與理解自評及原文家族留出是不同問題。 [MedReadMe §2、附錄 C/J](https://pmc.ncbi.nlm.nih.gov/articles/PMC12225841/) · [作者發布檔](https://github.com/chaojiang06/medreadme/tree/main/dataset)

Iavarone 等人的英文資料中有 943 個目標句在末尾的三句窗，讀者先讀可見上下文，再給難度評分。這符合因果停點可見範圍。素材取自 Wall Street Journal Penn Treebank，TextComplexityDE 的公開 Wikipedia 句子及文章署名較容易核實。Štajner 等人的五級資料則把同源句子的多個改寫版本相鄰呈現。以上兩個資源各可研究其特定可見條件。此次封存只有德語 Wikipedia 的兩個自評端點。 [英文上下文實驗 §2.1](https://aclanthology.org/2021.cmcl-1.23.pdf#page=3) · [Treebank 來源與授權](https://catalog.ldc.upenn.edu/LDC99T42) · [絕對句子複雜度 §3](https://ijcai.org/proceedings/2017/0572.pdf#page=2)
