# PoTeC 讀者背景留出可行性

PoTeC 的**數值反應與讀前背景**可直接使用 OSF 的 CC BY 4.0 發行來源。該來源足以評估先修背景對作答與逐詞眼動的統計對照。以完整教材正文在 Colab 做凍結推論或擬合仍缺本專案可核對的授權。 [原研究資料聲明](https://pmc.ncbi.nlm.nih.gov/articles/PMC12208991/) · [UZH 文本權利說明](https://cl.uzh.ch/en/research-groups/digital-linguistics/resources/potec.html) · [OSF 授權欄](https://api.osf.io/v2/nodes/dn5hp/)

## 原始發行檔與用途

| 檔案 | 核對結果 |
| --- | --- |
| 未合併的閱讀量測壓縮檔 | 900 個讀者×篇章 TSV、142,125 個逐詞列。欄位含 `reader_id`、`text_id`、`text_domain_numeric`、主修、修業階段、`sent_index_in_text`、`word_index_in_sent`、`FPReg`、`FPRT`、`RRT` 及各題正誤。欄位沒有教材表面字串。所有 900 個試次的主修、階段、expert 分組及六題正誤均與 GitHub 反應表一致。可直接從這份 CC BY 4.0 檔案抽取所需數值。SHA-256 為 `4119758d483f7a85a103169fa7b43609206fbd35ddb52b36d2d23e98e8945537`。 [OSF 數值檔](https://osf.io/download/g5jds/) · [原始欄位說明](https://osf.io/pke5q) |
| GitHub 題目與人員表 | 版本 `2742221e70cb9d2b38499bd080e079e6c0c1f5a3`。題目表含 12 個標題、36 道理解題、36 道領域題、各四個選項及正解。人員表可唯一連結 75 位讀者。反應表可按 `(reader_id, text_id)` 唯一連結 900 個試次，888 個有六題正誤，共 2,664 個可用的理解題反應。正式同讀者試次表只有正誤，選到哪個干擾選項無法取得。GitHub 沒有資料夾整體授權檔，題目文字與敘述選項如須傳到 Colab，向 DiLi Lab 核對研究重用範圍。數值正誤與分組已有 OSF 發行途徑。 [固定題目表](https://github.com/DiLi-Lab/PoTeC/blob/2742221e70cb9d2b38499bd080e079e6c0c1f5a3/stimuli/stimuli/stimuli.tsv) · [固定人員表](https://github.com/DiLi-Lab/PoTeC/blob/2742221e70cb9d2b38499bd080e079e6c0c1f5a3/participants/participant_data.tsv) · [固定反應表](https://github.com/DiLi-Lab/PoTeC/blob/2742221e70cb9d2b38499bd080e079e6c0c1f5a3/participants/participant_response_accuracy.tsv) |
| UZH 正文與出版社許可 | 12 篇 126–180 詞德文教材摘錄與固定題目表按 `text_id` 合併，雙方的共同欄位全相符。逐份核對 12 份 Springer Nature RightsLink 許可，皆授給 UZH 在其公開網站顯示指定摘錄，條款載明授權不可轉讓或再授權，2029 年到期。未見授權本專案上傳至 Colab 或作模型推論與擬合。OSF 的 AOI 檔有原文逐字元欄位，合併詞特徵檔也可能帶原文字詞，應保留相同文本權利判斷。正文及許可 ZIP 的 SHA-256 分別為 `3cce5828d377da1968f56421f92357016ea0f528dba0571f6d7da1cef2054ee3` 與 `3eb00e69d7cdf79ddd1fcac7015b49f08d41800aef2e830f8145011e66d16891`。 [UZH 正文與 12 份許可](https://cl.uzh.ch/en/research-groups/digital-linguistics/resources/potec.html#accessing_the_data) · [OSF AOI 檔](https://osf.io/download/jvqxa/) |

出版社提供訂閱機構非商業文字與資料探勘途徑，頁面要求材料存放於安全的內部伺服器並限制第三方存取。若要續做全文實驗，由機構確認訂閱範圍，或向 Springer Nature 索取涵蓋這 12 篇與私有 Google Colab 執行環境的書面同意。請分別說明只做凍結推論與擬合讀出頭的用途、雲端保留期限，以及只在本機保存真人數值反應、無正文重散布的安排。使用題目字串時另取得作者確認。 [Springer Nature TDM 條款與聯絡方式](https://www.springernature.com/gp/researchers/text-and-data-mining) · [書籍許可窗口](https://www.springernature.com/gp/policies/editorial-policies/third-party-permissions) · [PoTeC 作者團隊](https://cl.uzh.ch/en/research-groups/digital-linguistics/resources/potec.html)

## 讀者背景與端點

主要端點可用讀後三道理解題的二元正誤。受試者先讀可自由回看的完整單頁，再逐題作答，不能返回正文。領域題隨後才出現。主修及大學初階／研究所階段是收案時已有的條件，適合作為唯一的讀者輸入。人口問卷在實驗後填寫。領域題正誤、同讀者其他試次的理解分數及眼動都留在量測端。另有 35 位獨立讀者在沒看正文時作答 1,260 題，答對 330 題。這可檢查題目本身的可猜程度，與同一讀者的前測要分開。 [原研究 Procedure、Demographic questionnaire、Validation](https://pmc.ncbi.nlm.nih.gov/articles/PMC12208991/) · [無正文驗證資料](https://github.com/DiLi-Lab/PoTeC/blob/2742221e70cb9d2b38499bd080e079e6c0c1f5a3/participants/response_accuracy_online_survey.csv)

**數值對照可運作。** 以同一教科書卷為一組，7 組為 `b0`、`b1`、`b2`、`b3–b5`、`p0+p3`、`p1+p5`、`p2+p4`。以主修×修業階段分層，依 `SHA256("potec-reader-v1:" + reader_id)` 排序後循環分配五折。每一個書卷×讀者折的測試格，訓練只用其他書卷且其他讀者折。以 Beta(1,1) 平滑訓練格答對率，對 2,664 個測試反應做逐題等權的描述性比較。整體先驗、只用文本學科、使用文本學科×讀者主修×修業階段三者的折外 NLL 分別為 **0.65760、0.64477、0.63802 nats**。本次未對差值估區間。物理材料三卷屬同作者系列，因此這個切分驗證未見書卷與讀者，無法單獨建立未見作者或學科泛化。 [教科書來源表](https://github.com/DiLi-Lab/PoTeC/blob/2742221e70cb9d2b38499bd080e079e6c0c1f5a3/stimuli/STIMULI.md) · [數值反應原檔](https://osf.io/download/g5jds/)

若正文權利清楚，可在相同的雙重留出格比較三種模型。文本模型只看正文，背景模型看相同正文加主修與修業階段，結構對照使用學科、階段、匹配與篇幅。每個模型使用各條件指定的讀前資訊，沿用同一折內校準規則。目前的凍結候選只支援預設讀者，可先以其理解自評讀出檢驗整份 PoTeC 的外部排序，有背景的比較須待背景讀出固定。正誤預測若用訓練折進行校準，再報逐題機率和 NLL。附帶的逐詞 `FPReg` 可評估回視從哪個詞發起，`RRT` 可評估讀者後來在哪些詞停留。疑問內容、主觀負荷及解除時點需要相應的直接量測。 [PoTeC Table 12](https://pmc.ncbi.nlm.nih.gov/articles/PMC12208991/#Tab12) · [CogLLM 的原資料決策 D07](../stage-1/data-selection.md)
