# Qiao 與 Lagnado 的逐段判斷更新

兩項實驗追蹤**同一讀者在讀到新書面證據後，如何重評同一個指定命題**。公開資料每一觀測點都有 0–100 的有罪機率評分、二元裁決選擇、0–100 的信心欄及兩項條件機率。第一筆評分在讀到證據後才出現。完整證據後的裁決依然是讀者回應，刺激與觀測表皆沒有設定正確答案。沒有自發疑問、首次察覺時刻或疑問解除的紀錄。 [原文 Study 1 Methods、Study 2 Methods](https://escholarship.org/content/qt1zj2p06s/qt1zj2p06s_noSplash_1f0b500b4d2d2385f750b4aa8628058b.pdf) · [作者資料](https://osf.io/u72fr/)

## 資料與事件位置

| 樣本與呈現方式 | 作者讀者紀錄 | 每人觀測 | D→P／P→D 讀者 | 最終有罪評分 D→P／P→D |
| --- | ---: | ---: | ---: | ---: |
| Study 1，逐件 | 81 | 8 | 43／38 | 67.33／48.29 |
| Study 1，每組看完評分 | 78 | 2 | 38／40 | 54.95／49.98 |
| Study 2，先逐件後整組 | 93 | 5 | 45／48 | 60.67／55.35 |

D 是辯方，P 是控方。最終評分以每位讀者末次 0–100 分等權平均。Study 1 的 804 列使用作者的分析 ID，與其 R 程式一致。底層平台欄有兩個值各對到兩個分析 ID，故 159 筆作者讀者紀錄有 157 個不同的公開平台值。Study 2 工作表有 465 列、93 個不同的分析及平台值。兩研究的平台值無交集。論文記錄 Study 2 招募 95 人，其中 2 人因注意力題排除。 [Study 1 CSV](https://osf.io/download/whmzj/) · [Study 2 XLSX](https://osf.io/download/2y4pr/) · [兩研究程式](https://osf.io/u72fr/)

同一份材料 PDF 第 1 頁是案情，第 2 頁依方列出各 4 段具編號的證據，第 3–4 頁列出條件機率的提問。兩研究的兩種材料順序均保留同一方內部的 1–4 次序。Study 1 逐件組對齊到各段末，整組組只有第 4、8 段末。Study 2 對齊到第 1、2、3、4、8 段末。Study 1 一位讀者在第 1、5 次的來源證據標記與作者設計互換。由於其他停點也依賴前文的順序，該讀者八列保留原分數，文字停點全數留空。其餘 796 列的來源標記及軌跡與設計一致。Study 2 的表只有組別與階段，465 列的文字對齊依原論文的呈現設計。公開清冊未見 Qualtrics 檔，故可核對作者材料和設計次序，逐字畫面、標題是否重現與前期評分是否重顯仍未驗證。 [材料 PDF](https://osf.io/download/znkef/) · [原文 Procedure](https://escholarship.org/content/qt1zj2p06s/qt1zj2p06s_noSplash_1f0b500b4d2d2385f750b4aa8628058b.pdf)

Study 1 逐件組的 D→P 與 P→D 末次評分差為 +19.04 分，原文的組內比較 (p=.009)。整組組的原報告 (p=.983)。Study 2 的末次差為 +5.31 分，原文 (p=.280)。兩研究共用一個案件與八段證據，只能比較這個案件的呈現順序及提問時間。Study 2 提供不同讀者與新混合提示條件，沒有獨立文章可供留出。若評估文章單次因果前向，同一順序且讀到同段時，正文完全相同。要檢驗不同評分時程下的反應差，需在資料設計中提供讀者事前知道的評分時程，或事先限定單一時程。 [原文 Study 1 Main analysis、Study 2 Main analysis](https://escholarship.org/content/qt1zj2p06s/qt1zj2p06s_noSplash_1f0b500b4d2d2385f750b4aa8628058b.pdf)

## 檔案與重用

| 檔案 | OSF | SHA-256 |
| --- | --- | --- |
| Study 1 CSV | [whmzj](https://osf.io/download/whmzj/) | `f8c6ca10bda24d510e4f8f8f940faad054335dc0499b02100bc389ea5c3ac2e6` |
| Study 1 Rmd | [p9nfq](https://osf.io/download/p9nfq/) | `7c851861a8a3d7a67aeb1bcc2b1f0ef318e942e37ffadb30309d3be6a0f174c7` |
| Study 2 XLSX | [2y4pr](https://osf.io/download/2y4pr/) | `ff5aca229054500dbbe37a1e12a4d3294b6afc447cf99dce46f7c3c21cef6c6d` |
| Study 2 Rmd | [mn8xb](https://osf.io/download/mn8xb/) | `8f4b83e27b8b71ee01bc301b7c89f63e4e3445d98b4a6f96a38c6d2c6b54dade` |
| 材料 PDF | [znkef](https://osf.io/download/znkef/) | `31c660cb7a79d573ecf558f7d8f4c21a8cc3ec4fe28f500502e2531d0ca41b21` |

論文標示 CC BY 4.0。OSF 專案是公開的，專案授權欄未填，也沒有授權關聯。未找到可套用到獨立試算表、程式及材料的另行授權。可重現程式用固定版本的 PDF 擷取案情與證據，合併同段排版空白，移除標題與編號，使用排他字元座標。原始平台欄、作者 ID、自由作答和獨立正文都未提交至版本庫。兩次本機執行的輸出逐位元一致。 [論文授權](https://escholarship.org/uc/item/1zj2p06s) · [OSF 中繼資料](https://api.osf.io/v2/nodes/u72fr/) · [本機建置程式](qiao_evidence_prepare.py)
