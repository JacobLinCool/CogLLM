# Hypothesis 的公開提問、回覆與原提問者確認

Hypothesis 的公開資料可連起一則貼在正文上的問題、另一位使用者的回覆，以及原提問者針對該回覆的確認。這次找到兩則結構完整的討論，其中一則連到有明示開放授權的教科書章節。可驗證的時間由三則發文的時間戳決定，能觀察貼文與回覆的順序。發文時讀者已看過正文哪些部分，需要另一份曝光紀錄。 [Hypothesis 公開 API](https://web.hypothes.is/developers/) · [公開標註 CC0 說明](https://web.hypothes.is/help/what-is-the-license-on-annotations/)

## 兩種分母

先在 `GET /api/search` 固定 `group=__world__`，以確認短語找可能的案例。`any=thanks for clarifying` 共回傳 18 筆，其中 12 筆是帶 `references` 且正文確實包含這個短語的回覆。兩筆同時有標到正文的原問題、另一帳號的直接回覆、原帳號的後續確認，同一 URI 及正向時間順序皆核對一致。這是依確認措辭找案例，無法推估一般提問被回答的比例。 [API 查詢方式](https://web.hypothes.is/help/how-hypothesis-search-works/)

再各按精確 URI 取回該頁所有公開紀錄，這一步未用確認短語篩選。原問題候選依根留言的疑問措辭與 `TextQuoteSelector`、`TextPositionSelector` 選出，並計數有其他帳號回覆、原帳號回覆另一人，以及後者包含固定確認短語的根留言。兩個主要頁面由上述案例發現，另查教科書的相鄰兩節。 [計數與條件](../results/hypothesis-clarification-source-audit.json)

| 文章 URI | 公開紀錄 | 根留言 | 有文字錨點及疑問措辭 | 有另一人回覆 | 原提問者跟進 | 有固定確認措辭 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Open RN 3.5 | 24 | 22 | 20 | 1 | 1 | 1 |
| Open RN 3.4 | 0 | 0 | 0 | 0 | 0 | 0 |
| Open RN 3.6 | 2 | 2 | 2 | 0 | 0 | 0 |
| Marginal Syllabus PDF | 273 | 186 | 36 | 12 | 3 | 1 |

## 核對範圍

- **Open RN 3.5。** 原章節明載 CC BY 4.0，另有注明的例外。五字元的選取長度與位置選擇器一致。這段文字在同版 NCBI 章節的空白正規化文字中只出現一次，但連帶選擇器的前後字串未完整吻合。原版 HTML 及官方 XHTML、XML 匯出請求均回傳 HTTP 403，原始 DOM 字元座標仍待逐字核對。 [出版商章節與授權](https://wtcs.pressbooks.pub/medterm/chapter/3-5-diseases-disorders-and-injuries-of-the-integumentary-system/) · [NCBI 同版章節](https://www.ncbi.nlm.nih.gov/sites/books/NBK607449/)
- **Marginal Syllabus。** 原問題的文字選擇器與位置寬度均為 15 個字元。三則留言的公開授權已確認。該 PDF 的獨立重用條款及原文偏移未查清，保留為互動資料的證據。 [專案與資料類型](https://education.ucdenver.edu/partnerships/our-impact/marginal-syllabus) · [原 PDF](https://educatorinnovator.org/wp-content/uploads/2018/11/What_s_Radical_marginal-syllabus.pdf)

若找到更多授權與原文均可核對的案例，可測試給定同一個問題和正文錨點時，模型能否將原提問者確認的回覆排在同文其他問題的回覆之前。`references`、`created` 與帳號相同性可建立已觀測的三回合關係。未留言或未再回覆的讀者，其理解狀態未被觀測。這份稽核目前只確認兩則公開互動的連接方式，完整閱讀的停點、群體解除率與逐 token 解除仍缺相應觀測。
