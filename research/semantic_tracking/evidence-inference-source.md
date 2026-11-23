# Evidence Inference 2.0 的跨領域證據定位

Evidence Inference 2.0 提供完整臨床試驗論文與醫師標註的證據區間，可檢驗同一正文換一個已知查詢時，分數是否移到另一組人工選取的位置。原研究先由醫師撰寫以介入、比較對象及結果為條件的查詢，再由另一位醫師閱讀正文、作答並選取證據，最後由第三位醫師審核答案與證據。實驗端點限於正文中報告的結果及離線選取的位置。發布檔缺少首次發現證據的時間、閱讀期間的疑問狀態，以及同一讀者在不同閱讀停點的回報。 [原研究 §2.1–2.3](https://aclanthology.org/2020.bionlp-1.13.pdf)

## 來源與授權

核對作者 GitHub 的 `a661e8c14f973398380c8865cf2f27a535aaaf6d`。以下路徑相對於該版本的 `annotations/`。以 `(PMCID, PromptID)` 串接查詢與證據，以 `UserID` 區分標註者。字元座標以作者同版 `txt_files/PMC<id>.txt` 為準，文章授權由 `xml_files/PMC<id>.nxml` 的原始權利聲明核對。作者亦在 Hugging Face 以 MIT 標示資料集。PMC 原始文章的條款各自獨立，進入評估前須逐篇留存明示 CC BY、CC0 或公有領域的授權證據，並檢查第三方素材的例外。 [固定版本檔案](https://github.com/jayded/evidence-inference/tree/a661e8c14f973398380c8865cf2f27a535aaaf6d/annotations) · [作者的 Hugging Face 發布](https://huggingface.co/datasets/jaydeyoung/evidence_infer_treatment) · [PMC 開放取用條款](https://pmc.ncbi.nlm.nih.gov/tools/openftlist/)

| 檔案 | 發布檔筆數 | SHA-256 |
| --- | ---: | --- |
| `annotations_merged.csv` | 24,686 筆判定 | `a5dd1a3583e5105a0f77d3e1c695008879859425c9a93dfbc07054ef462e0255` |
| `prompts_merged.csv` | 12,865 筆查詢 | `0d70d9e1e78d113fdbbd4919310eab7768d6a328368f7d8f9d5a3b48f99c25d3` |
| `splits/test_article_ids.txt` | 449 篇論文 | `4fe6c633ba8f077a68a8feea48d4408c1749175e1c46f2b77d96727d291ef9d8` |

## 已核對的位置

每列兩個查詢的全部已核准證據區間皆互不重疊。依 UTF-8 解碼後的正文用半開位置 `[start, end)` 取字元，逐一與發布檔的原引文完全相同。數位相同的區間可能由多位醫師各自選取，表內每種區間只列一次。 [作者欄位及正文說明](https://github.com/jayded/evidence-inference/tree/a661e8c14f973398380c8865cf2f27a535aaaf6d/annotations#article-types-xml-vs-txt)

| 官方測試 PMCID | 空白切分詞數 | 文章授權 | 查詢 A 與區間 | 查詢 B 與區間 |
| --- | ---: | --- | --- | --- |
| `PMC1871574` | 5,648 | CC BY 2.0 | `286`，`[17327,17629)` | `298`，`[19921,20317)` |
| `PMC3536709` | 6,114 | CC BY 2.0 | `3166`，`[17080,17192)` | `3169`，`[17429,17524)` |
| `PMC4497728` | 5,459 | CC BY 4.0 | `8982`，`[1711,1987)` | `8988`，`[2536,2658)` |

初篩在 275 篇官方測試論文找到 1,711 組同文配對。初篩只用獲驗證者同時核准答案和理由、區間可解析且有引文的列，要求兩個查詢在保留的區間間完全分離。這是正式篩選前的可用性估計，尚未逐篇核對授權與全部引文。作者另外公布有問題的 PromptID，並記錄部分區間為 `-1` 或偏移數個字元。正式匯入須先固定排除規則，再要求每個採用的區間逐字對齊。欄位說明把終點寫為含端點，上述實際資料則與不含端點的切片完全吻合。 [作者的座標修訂與問題清單](https://github.com/jayded/evidence-inference/tree/a661e8c14f973398380c8865cf2f27a535aaaf6d/annotations#caveat)

文章長度應用實際 tokenizer 測量，並計入查詢與分隔文字。上表的空白切分詞數無法判定完整正文能否放進現有 8,192 token 預算。正式測試前還須固定可從正文事先取得的候選邊界，核對人工字元區間映射與因果前綴的 token 邊界，記錄長度或對齊造成的涵蓋率。上述三篇已證明同文異證據的配對存在。這份來源稽核完成於模型測試之前。
