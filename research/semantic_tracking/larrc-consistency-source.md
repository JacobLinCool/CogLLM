# Currie 等人的篇末一致性判斷

Currie 等人（2021）讓六年級學生逐句閱讀八句英文文章，讀完才回答文章是否說得通。公開資料可用來檢驗模型對**篇末明示一致性判斷**的讀出，涵蓋敘事文和動物主題說明文。矛盾句所在的句號位置可以由刺激材料指定。真人判斷的時間點是全文結束，同一次閱讀中首次察覺的句子未觀測。 [原文 Measures and procedure、Table 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC7986958/) · [作者存檔](https://osf.io/sj28g/)

## 公開檔案與重用

作者的 OSF 資料及刺激子專案都明列 CC BY 4.0。以下雜湊以下載檔重算。授權來自作者存檔，期刊論文本身採 CC BY-NC-ND 4.0。 [刺激授權欄](https://api.osf.io/v2/nodes/vp45x/) · [資料授權欄](https://api.osf.io/v2/nodes/g9pyc/) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [期刊權利說明](https://pmc.ncbi.nlm.nih.gov/articles/PMC7986958/)

| 作者檔案 | 位元組 | SHA-256 |
| --- | ---: | --- |
| [CM Narrative Materials_OSF.docx](https://osf.io/download/b73ks/) | 41,397 | `1c62b97572d0a671fd0f7780815328782553107b894bb356b07f7cac39400082` |
| [CM Expository Materials_OSF.docx](https://osf.io/download/3pswr/) | 53,480 | `27b085c6be95d165484a002f1b4b13ddd6322d104535dcb349406305fb3fae48` |
| [cm.data.and.TOWRE.csv](https://osf.io/download/swjdf/) | 510,504 | `9f649f59e1eb517a562eb398979bb185d0cba81415da87ba8602682bddda2f85` |
| [Variable descriptions for CM paper_OSF.xlsx](https://osf.io/download/tq36h/) | 10,877 | `c76c2cadf4d46eeb4b2b22d64ec242987dcb2cb1f6eb3e1447f1237155bef204` |

## 刺激與作答對齊

- 每種文體有 12 組矛盾文本。DOCX 以 `1` 至 `12` 編號，每組在 `NEAR`、`FAR` 標頭下列出各八句。CSV 的 `Text_Type` 對應文體，`Test` 對應組號，`Condition` 對應版本。`text` 是 `Test` 與 `Text_Type` 的串接值。兩版共用同一組號。受試者每組只看一版，兩個版本的差異由不同受試者的回報估計。 [刺激檔](https://osf.io/vp45x/) · [對齊欄位定義](https://osf.io/download/tq36h/) · [原文 Counterbalancing](https://pmc.ncbi.nlm.nih.gov/articles/PMC7986958/)
- 兩種文體各有另寫的六篇一致文本。CSV 編號 `13` 至 `18`，條件為 `consistent`。DOCX 在各自的 `Consistent items` 小節依序列出六篇。說明文重新從 `1` 到 `6` 編號，敘事文只列標題。可按作者檔的排列順序建立這 12 個對應。公開存檔缺少可獨立核對順序的 E-Prime 題號索引。使用一致文本做逐篇評分需揭露這項對齊假設。 [作者刺激檔](https://osf.io/vp45x/) · [作者資料清冊](https://osf.io/g9pyc/)
- 去除空白段落、題目標頭、練習題、問答欄與一則作者編輯註後，得到 60 個不同的八句版本。`Birds of paradise` 的 `FAR` 版以行首手寫數字標示句序，其餘列表編號為 DOCX 排版。比較時移除前者的八個編號。矛盾版本用底線標出相衝突的兩句。`Hannah` 的 `NEAR` 版另有底線格式落在空白，取有文字的標記即可。 [敘事 DOCX](https://osf.io/download/b73ks/) · [說明 DOCX](https://osf.io/download/3pswr/)
- 24 組近遠配對中，23 組的目標句相同，`Hannah` 的兩版目標句有逗號差異。正規化大小寫、曲撇號及空白後，以完整句的多重集合比較，20 組只有順序不同。其餘四組含標點、措辭或指涉表達差異。只檢驗句序影響時，20 組可預先以文字規則選出。依同一正規化程序，在 372 筆 Frick 推論文字中搜尋這份資料的 294 個不同完整句，找到零筆完整句重合。 [近遠刺激](https://osf.io/vp45x/) · [Frick 材料來源](frick-consistency-source.md)

## 可評真人結果

CSV 有 6,372 個有效二元回報，179 位讀者與 36 個原始題材。`(Subject, Text_Type, Test)` 在全集唯一。175 位讀者交齊兩種文體的 36 題，另有四位交齊其中一種文體的 18 題。共有 178 人完成敘事文、176 人完成說明文。每個版本有 87 至 178 筆回報。60 個版本各自同時有正答與誤答，179 位讀者也各自同時有正答與誤答。 [逐試次資料](https://osf.io/download/swjdf/) · [原文 Measures and procedure](https://pmc.ncbi.nlm.nih.gov/articles/PMC7986958/)

`Question.ACC` 記的是篇末一致性回答是否正確。對近、遠矛盾文本，答對表示回報文章有矛盾。對一致文本，答對表示回報文章一致。若讀出目標定為「回報矛盾」，一致條件應取 `1 − Question.ACC`，矛盾條件直接取 `Question.ACC`。下表為資料重算的正答數與可評分總數。 [欄位定義](https://osf.io/download/tq36h/) · [原文 Measures and procedure](https://pmc.ncbi.nlm.nih.gov/articles/PMC7986958/)

| 文體 | 一致 | 矛盾較近 | 矛盾較遠 |
| --- | ---: | ---: | ---: |
| 敘事文 | 910 / 1,068 | 696 / 1,068 | 693 / 1,068 |
| 說明文 | 867 / 1,056 | 423 / 1,056 | 431 / 1,056 |

這組資料可獨立評估篇末回報，包括同篇調整句序後的差異。分割資料時兩版按同一題材分組。一致條件的題材和矛盾條件不同，合併比較時要保留此設計差異。Frick 的「讀到目標句立即按 `D`」有另一個觀測時間點。譬如學生在目標句當下感到違和，讀完卻回答文章說得通，這份資料只會留下篇末回答。 [Currie 等人的呈現及提問順序](https://pmc.ncbi.nlm.nih.gov/articles/PMC7986958/) · [Frick 的逐句作答流程](frick-consistency-source.md)

上述四個檔案共 616,258 位元組。來源對齊可在 CPU 完成。若後續為 60 個版本的每個句尾各讀出一個 2,048 維 FP32 向量，480 個位置的陣列本體約 3.75 MiB。這是容量估算，此次工作只核對文本與觀測，尚無模型實驗結果。
