# ACT 的真人首次答對位置與線索對齊

Löffler 與 Topolinski 的德文 ACT 主實驗公開了 **100 位讀者 × 20 份詞表**的結果。每份詞表有固定編號的 15 個線索，讀者每次看到一個詞，之後可輸入猜測或略過。答對後立即結束該題。兩批各 100 人的前導研究用來評估線索接近程度，未併入首答分析。 [原研究，Pre-study、Sample、Materials and Procedure](https://reference-global.com/article/10.5334/joc.254?tab=article) · [作者公開資料](https://osf.io/et5kb/)

## 已核對的材料

從 OSF 的詞表 PDF 取出 20 個解答標題及 300 個編號線索。期刊附錄與 OSF 檔案的 PDF 位元不同，逐項核對得到完全相同的標題、詞形和順序。每個模型輸入保留原線索拼寫與順序，用換行連接。已排除標題、答案、接近程度評分和試次資料。在第 7 份詞表，第七和第八個線索的附錄接近程度同值，重建依兩個來源一致的編號保留原順序。這個同值在六步主分析以外。公開資料提供編號和步數，沒有逐試次的畫面播放紀錄。 [期刊附錄](https://doi.org/10.5334/joc.254.s1) · [PMC 保存的原附錄檔](https://pmc-oa-opendata.s3.amazonaws.com/PMC9838243.1/joc-6-1-254-s1.pdf) · [OSF 詞表原檔](https://files.us.osf.io/v1/resources/et5kb/providers/osfstorage/6284cc9c47df854ff26ef906)

德文說明頁先展示一份答案已知的例子。正式詞表順序在各讀者內重新隨機化。每份詞表內按既定順序呈現線索，各顯示兩秒。略過的步驟仍算一次線索曝光。作者作答檔的 `clues_*` 與 `unsolved_*` 欄能逐項接回 20 份原詞表。檔案未保留讀者的逐步原始猜測與個人化詞表次序。 [德文指示原檔](https://files.osf.io/v1/resources/et5kb/providers/osfstorage/6380c985f284991927ddcce1) · [作答原檔](https://files.us.osf.io/v1/resources/et5kb/providers/osfstorage/62666ba85bb1ac1b2d460e95)

| 評分範圍 | 可定位的首次答對 | 到期仍未答對 | 作者標為答對且缺失步數 |
| --- | ---: | ---: | ---: |
| 前六次曝光 | 355 | 1,644 | 1 |
| 全部十五次曝光 | 1,321 | 678 | 1 |

第六步後答對的試次在主分析視為六步右設限。缺失步數的試次獨立保留，無法從個人總覽欄恢復，兩種分析的可評分母都是 1,999。可定位的首次答對事件對應線索詞。與一個詞拆成哪些子 token，以及普通文章中的自發疑問何時形成，分別有不同的可觀測粒度。 [刺激與結果重建](act_prepare.py) · [端點與模型比較](act-transfer-protocol.md)

期刊文章標示 CC BY 4.0，獨立 OSF 資料檔的公開後設資料未標示授權。重現程式釘選原檔雜湊。原始人類反應、去識別試次與全文詞表都保留在本機的重現資料夾。 [文章授權和資料可取得性](https://pmc.ncbi.nlm.nih.gov/articles/PMC9838243/) · [OSF 專案資料](https://osf.io/et5kb/)
