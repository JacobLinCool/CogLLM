# CommonLit CLEAR 英文篇章閱讀容易度來源稽核

CLEAR 可用來檢驗固定 App 候選對英文篇章的容易度排序。原研究以國小三年級至高中十二年級的學生閱讀為情境，請教師在兩段文字中選出較容易讓學生理解的一段。研究保留 1,116 名教師的 111,347 次配對判斷，將判斷整合成每篇一個 Bradley–Terry 估計。公開表的 `BT_easiness` 越高表示預期越容易理解。這個相對尺度沒有對應到年級，公開檔也沒有逐次選擇、評分者或句級判斷，無法量測學生實際理解表現、個別教師的判斷或閱讀困難出現的位置。原研究將判斷試次隨機分成兩半並各自估計分數，兩個半組的相關為 0.63。 [Crossley 等人的原始研究，Study 1](https://link.springer.com/article/10.3758/s13428-022-01802-x)

## 固定發布版與欄位

作者 GitHub 提交 `dbf1cf0df30bf6be8b0691d2c078b75043ca6199` 的 `CLEAR_corpus_final.xlsx` 大小為 3,349,179 bytes，SHA-256 為 `e87a19a29c12a441e0c4f11a4e49897ff24ac7dff240ee7397283a050a902cc7`。同一位元組檔案已存放在忽略追蹤的 `.cache/cogllm/clear-reading-ease/raw/CLEAR_corpus_final.xlsx`。工作簿有 `Data` 和 `Data Key` 兩張表。`Data` 含 4,724 篇、28 欄，`ID` 在 Excel 中為數值且全數唯一。每個 `Excerpt` 都包含教師當次看到的完整文字，其中 122 篇的 `Location` 為 `whole`，指原作品全文。其餘為作品摘錄。`BT_easiness` 和 `s.e.` 均無缺值，範圍分別約為 −3.676 至 1.711，及 0 至 0.650。`s.e.` 是每篇容易度估計的標準誤，發布版沒有配對共變異或逐次標籤。 [固定工作簿](https://raw.githubusercontent.com/scrosseye/CLEAR-Corpus/dbf1cf0df30bf6be8b0691d2c078b75043ca6199/CLEAR_corpus_final.xlsx) · [作者發布目錄](https://github.com/scrosseye/CLEAR-Corpus/tree/dbf1cf0df30bf6be8b0691d2c078b75043ca6199)

## 重用範圍

固定 GitHub 版 README 指定資料採 CC BY-NC-SA 4.0。CommonLit 發布說明另對其發布版的資料與中繼資料指定 MIT，並分別列明摘錄權利。本次採固定 GitHub 版的非商業研究條件，另核對每列的 `License`。留出測試只用明示且版本完整的單一 CC BY 或 CC BY-SA 列，重複及跨材料重疊排除前至多 1,016 篇。公開結果可僅含座標、雜湊、中繼資料及統計數值。空白授權列的個別權利狀態未確認。 [作者發布版授權](https://github.com/scrosseye/CLEAR-Corpus/blob/dbf1cf0df30bf6be8b0691d2c078b75043ca6199/README.md) · [CommonLit 發布與授權說明](https://www.commonlit.org/blog/introducing-the-clear-corpus-an-open-dataset-to-advance-research-28ff8cfea84a/#licensing-and-use-restrictions)

| `License` 列值 | 篇數 |
| --- | ---: |
| 空白 | 3,319 |
| `CC BY 4.0` | 694 |
| `CC BY 3.0` | 16 |
| `CC BY-SA 3.0` | 306 |
| `CC-BY-SA`（版本未註） | 1 |
| `CC BY-SA 3.0 and GFDL` | 274 |
| 含 NC 的各版本 | 113 |
| `GNU Free Document License, version 1.2` | 1 |

## 分組、重疊與可檢驗端點

- 原文完全一致的摘錄有 0 對。以 Unicode NFKC、casefold 及合併空白正規化後有 1 對重複。初步來源稽核將 URL 網域移除 `www.`，並將路徑解碼、去除尾斜線與 fragment，保留 query。此法取得 2,689 個非空 URL 群，218 群有多篇。正規化作者及標題取得 4,679 個鍵，35 個鍵有多篇。相同來源 URL、同作者及標題、或高度重疊正文應併為連通元件，再以元件切分及重抽。正式重複及來源分組規則見[實驗規格](clear-reading-ease-protocol.md)。
- 對 4,724 篇進行初步跨資料核對。與目前的 PLABA 78 個、CompLex 9,165 個、IIRC 5,183 個及 WikiNews 591 個輸入做正規化全文比對，命中為 0。對至少 12 個空白詞的既有輸入，檢查其整段原文是否完整出現在 CLEAR 摘錄中，命中亦為 0。後續封包另依實驗規格核對局部共同正文，這兩項初步結果的範圍限於已列出的輸入，模型預訓練文字未知。
- 封存固定候選與表面基線後，可將 App 已有的逐句分數依原文詞數彙整成篇章分數。以留出來源的 Spearman 相關衡量模型排序與教師的篇章估計是否一致，配對重抽來源元件來估計差異。公開檔支持這個篇章排名端點，句級位置、歧義出現與解除時間、真人作答正確率均需獨立證據。 [實驗規格](clear-reading-ease-protocol.md) · [原始研究的排名與語料說明](https://link.springer.com/article/10.3758/s13428-022-01802-x)

## 正式材料準備

1,016 篇符合這次的授權條件。排除一篇重複、一篇缺可用來源鍵及四篇 App 句末無法完全對齊的材料後，留存 **600 篇校準及 410 篇留出**，分別來自 596 和 405 個不重疊的來源元件。初步來源群組把同一出版平台的數百篇不同故事併在一起，正式切分改用上面規定的資源 URL、作者標題和重疊正文，成員最多的元件有三篇。在五份列入收據的既有材料檔中有一篇與 IIRC 分享一段 12–29 詞文字，位於校準組。未發現達 30 詞的重疊。兩次建置逐位元相同，12,743 個句末均用 App 的分詞定位入口再次核對。材料與教師估計分開存放，這次只完成 CPU 準備。 [準備程式](clear_reading_ease_prepare.py) · [結果、分母及輸入雜湊](../results/clear-reading-ease-preflight.json)
