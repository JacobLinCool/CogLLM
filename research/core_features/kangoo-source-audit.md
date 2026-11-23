# 英文教材中的真人難詞圈選

Kangoo 等人以英文教材做難詞圈選，參與者是 767 位大學生及 193 位研究生。紙本作業要求在原文下畫線，線上作業先呈現同類段落，再用詞彙核取方塊作答。公開票數合併兩種方式，也沒有保存個別參與者鍵。這份資料量到的是一個詞被標成困難的次數。每頁實際完成作答的人數無法由公開資料重建。 [論文 §§3–4](https://pmc.ncbi.nlm.nih.gov/articles/PMC10661753/#sec0004) · [作者固定發布 v4](https://data.mendeley.com/datasets/p2wrs7hm4z/4)

已驗證作者發布的十份問卷和 `dataset_english.csv`，共約 473 KB。十份 Word 檔各有九個既有的分頁標記，每份可重建十個來源頁面。保留頁內文字、換段及定位用字元，沿用原始問卷分組。整批有 100 頁、29,880 個介面詞。可重現程式會先核對每個發布檔的 SHA-256。 [來源與解析程式](kangoo_prepare.py)

CSV 曾將 100 頁合併後依詞形去重。只在 CSV 頻次為 1、正規化詞形在 CSV 唯一且在全部問卷唯一對到一個完整詞時，才把詞形票數接回原文。該詞必須有其所屬大學或研究生群組的票數，另一組須標明未見，票數也須落在原作者所報該學期人數內。5,368 列中有 2,808 列通過，449 個詞至少有一票。55 個頁面內有票數不同的詞對，共 10,687 對，分屬十份問卷。這是可對齊樣本依據，尚未執行任何模型評分。 [詞形彙整與群組人數](https://pmc.ncbi.nlm.nih.gov/articles/PMC10661753/#sec0004) · [對齊規則](kangoo_prepare.py)

就專案現存的 CompLex 三個切分、CWI 2018 英語正式切分，以及已評的 MedReadMe 留出，比較正規化詞序後，沒有找到五詞以上的完整來源句嵌入問卷頁面。也沒有任何來源句在單一問卷頁面匹配到至少 25% 的相異五詞片段。Qwen 預訓練資料是否曾見過這些教材仍未知。

Mendeley v4 標示 CC BY 4.0。問卷引用的實際教材另有原出版社聲明。已確認至少兩份含相同問卷文字的 IGNOU 教材載明保留全部權利並要求書面重製許可。校方的開放教育資源政策允許用指定授權釋出教材，仍須逐個來源確認。本次接合只在本機快取重現，適用於來源稽核。 [Mendeley 資料授權](https://data.mendeley.com/datasets/p2wrs7hm4z/4) · [IGNOU 開放教育資源政策](https://www.ignou.ac.in/viewFile/pdd/policy-document/Institutional-OER-Policy-2022.pdf) · [BEGC-103 原出版社權利頁](https://egyankosh.ac.in/bitstream/123456789/64615/3/Block-1.pdf#page=2) · [BEGC-104 原出版社權利頁](https://egyankosh.ac.in/bitstream/123456789/66101/1/Block-2.pdf#page=2)
