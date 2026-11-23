# CogLLM 人類閱讀的理解與處理成本文獻

整理日期為 2026 年 11 月 20 日。這份目的性文獻庫收錄 **46 篇以人類反應為依據的實證論文**，其中 36 篇屬 2000–2026 年主要範圍，10 篇為 1980–1999 年的經典實證。2000 年是本專案選擇的檢索範圍。另列出理論、測量與閱讀媒介的參考文獻，以及可用來建資料或重做分析的資源。逐篇觀察、來源查核和模型用途見 Stage 1。 [觀察登錄](stage-1-observations.md)

檢索從跨句連貫、先備知識與閱讀理解研究擴展到心理語言學和篇章語言學。使用的詞群包括 `cognitive load`、`mental effort`、`working memory load`、`reading processing load`、`reading comprehension`、`discourse comprehension`、`bridging inference`、`coreference`、`discourse connectives`，並以「因果推論」「篇章理解」「指涉」找華語研究。早期經典依研究問題和引文追蹤補充。納入時優先選擇能辨認讀者、文本對照及觀測端點的原始研究。這是一份選題用的文獻庫，涵蓋率未經系統性評估。

文獻庫提供 **候選的閱讀效應與實驗對照**，包括觀測位置、受試者與條件差異。首批優先涵蓋局部概念、篇章關係與推論。答題、遷移、主觀心智努力和眼動各有自己的測量單位。理解作答可以校準同類題目的答錯風險。心智努力使用對應的讀者評分。每列為論文層級的證據摘要。具備可用素材、試次資料或可比較統計量的研究，才依取得的證據安排訓練、驗證或重做實驗。

## 2000–2026 核心實證

### 理解、推論與文本修補

| 研究 | 設計與觀測 | 對 CogLLM 的用途 |
| --- | --- | --- |
| **R04** Gilabert, Martínez & Vidal-Abarca（2005） | 西班牙中學生和大學生讀俄國革命相關文章。實驗 1 因果解釋版的開卷推論題表現優於原版。僅增加概念重疊的效果依實驗而變。 | 比較補齊因果前提與重複關鍵詞，保留閱讀和作答時序。 [§2–3、Appendices](https://doi.org/10.1016/j.learninstruc.2004.12.003) · [兩實驗詳錄](stage-1/coherence-and-knowledge.md) |
| **R05** Van Silfhout, Evers-Vermeul & Sanders（2015） | 荷語八年級生閱讀有／無連接詞的敘事和說明文。連接詞條件的局部橋接推論題表現較佳，後續區段讀得較快。 | 設計兩句之間的關係顯化對照，將語速與答案分別評估。 [Results、Discussion](https://doi.org/10.1080/0163853X.2014.905237) |
| **R06** Kendeou & van den Broek（2007） | 英語大學生先測牛頓定律迷思，再閱讀有／無明示反駁的文本。知識狀態與文本結構的交互作用出現在定律句的每音節閱讀時間。回憶有知識差異，文本結構效應未達顯著。 | 針對特定迷思及定律句測試，並保留回憶結果的統計不確定性。 [Experiment 2，Tables 2–3](https://link.springer.com/content/pdf/10.3758/BF03193491.pdf#page=7) |
| **R07** Gu, Leroy & Kauchak（2017） | 成人閱讀來自 Wikipedia／PubMed 的英語醫學片段。Study 1 中，技術定義與白話解釋條件的內容題答對率分別為 62.6% 和 72.1%，差 9.5 個百分點。篇章閱讀時間與該研究的理解結果未觀察到顯著相關。 | 用解釋版本與同一理解題作為修補對照。效應屬該研究的片段與受試者。 [Study 1，Results](https://pmc.ncbi.nlm.nih.gov/articles/PMC5977593/) |
| **R08** Cohen et al.（2025） | 重分析 R07 的成人作答。LLM 以句子、上文和摘要估計連貫性，部分分數與篇章答對率相關，並隨一些解釋改寫而改善。 | 可當連貫性方法基線。與 R07 共享資料，編輯位置已指定。 [§3.1、§4.2、Table 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC12243615/) |
| **R09** Tibken & Tiffin-Richards（2025） | 96 名德語讀者讀 20 篇短說明文。版本間改動一個詞以製造矛盾，讀後確認有無矛盾及所在句。目標詞的重讀與正確偵測相關，首遍訊號的支持較弱。 | 任務有明示矛盾偵測指令。可用公開試次表查驗正確性端點，研究中讀者所選句的索引未出現在該表。 [Methods、Tables 2–3、Discussion](https://doi.org/10.1007/s11409-025-09440-2) · [資料與程式](https://osf.io/ekf4t/?view_only=1cd488d47c254163bfc032fbbc7fa0cc) |
| **R10** Beker, Jolles, Lorch & van den Broek（2016） | 荷語大學生讀完前篇後讀第二篇。第二篇目標句一致，只有前篇是否交代可解決矛盾的解釋不同。有前置解釋時目標句讀得較快，實驗 2 自由回憶未觀察到對應差異。 | 固定目標字面內容、一併測量後續句與回憶。 [Experiments 1–2，Table 2](https://link.springer.com/article/10.1007/s11145-016-9630-3) |

### 讀者知識、資訊組織與閱讀目標

| 研究 | 設計與觀測 | 對 CogLLM 的用途 |
| --- | --- | --- |
| **R12** McNamara（2004） | 比較自我解釋閱讀訓練與朗讀控制。對低先備知識者，有絲分裂後測的訓練優勢出現在文本依據題，橋接推論題未呈現同樣效果。 | 約束訓練效益的適用題型。 [Results，Posttraining text、Fig. 1](https://doi.org/10.1207/s15326950dp3801_1) |
| **R14** Ozuru, Dempsey & McNamara（2009） | 美國大學生讀高／低銜接性的生物文本，分開作答文本、局部橋接、全局橋接題。高銜接性改善文本依據題，閱讀能力較高者在該研究中受益更多。先備知識與整體理解正相關。 | 訓練標註記錄題型與要求的證據範圍，並留出跨讀者群體檢驗。 [§2–3 與作者機構摘要](https://doi.org/10.1016/j.learninstruc.2008.04.003) · [作者機構](https://asu.elsevierpure.com/en/publications/prior-knowledge-reading-skill-and-text-cohesion-in-the-comprehens/) |
| **R16** Tibken, Richter & Wannagat（2024） | 較年輕與較年長的德語成人讀帶有矛盾操弄的說明文，指出是否有矛盾及出現在哪一句。以偵測個數及個人差異比較兩年齡組。 | 真實讀者背景的延伸任務。與 R09 跨研究比較前，先核對素材重疊。 [Methods、Fig. 2、Data availability](https://doi.org/10.1080/10888438.2023.2261572) |
| **R17** Wiley & Rayner（2000） | 為同一英語正文提供主題標題。實驗 1 的標題組在目標名詞和句尾的閱讀較快、回視較少，理解較佳。 | 檢驗上游主題提示如何影響下游歧義名詞，避免將標題本身當成唯一難點。 [Experiment 1](https://link.springer.com/content/pdf/10.3758/BF03209349.pdf) |
| **R18** Hyönä & Lorch（2004） | 成人讀兩篇多主題說明文，操弄有無小標題。小標題組的主題句首遍和回看處理較順，摘要涵蓋較多主題。 | 用同文的小標題版本評估資訊組織效果。 [原始研究，Results](https://doi.org/10.1016/j.learninstruc.2004.01.001) · [作者機構摘要](https://scholars.uky.edu/en/publications/effects-of-topic-headings-on-text-processing-evidence-from-adult-/) |
| **R20** Bai et al.（2026） | 新收集的英語閱讀實驗比較 30、60、90 秒限時條件。時間增加時，跳讀變少、回視變多，理解答題和自由回憶亦改善。 | 文章相同時，閱讀目標和時間限制仍構成實驗條件。可用資料測策略與結果的分離。 [Fig. 3e、Methods](https://doi.org/10.1038/s41562-026-02534-0) · [資料](https://osf.io/q2dm6/) |

### 定位與測量的邊界

| 研究 | 設計與觀測 | 對 CogLLM 的用途 |
| --- | --- | --- |
| **R22** Wang et al.（2006） | 中文逐行自定步速閱讀。在局部合理的情況下，目標內容與較遠背景抵觸時，比無關對照更慢。此模式在有無句法邊界的版本均出現。 | 中文跨句對照可固定局部合理性，再改動遠端前提。 [Experiment 1，原刊](https://journal.psych.ac.cn/xlxb/CN/Y2006/V38/I05/645) |
| **R23** Troyer et al.（2024） | 分開測量讀者對特定哈利波特事實的答題表現及整體領域知識，讀正確答案的關鍵詞時量測 ERP。即使讀者答不出該事實，高領域知識仍與較小的 N400 振幅相關。 | 概念級知識與領域熟悉度各自編碼。生理反應一般化到理解題的程度需要驗證。 [§1.2、Fig. 4、Table 4](https://doi.org/10.1111/psyp.14422) |
| **R24** Schotter, Tran & Rayner（2014） | 英語讀者讀有／無句法歧義的句子。遮罩阻止有效重讀時，理解問答表現下降，影響也出現在無歧義的句子。 | 回讀可以是修復策略，理解失敗模型要保留這個混淆。 [Results、General discussion](https://doi.org/10.1177/0956797614531148) |
| **R26** Demberg & Sayeed（2016） | 德語自定步速閱讀中，語意違反使目標詞較慢。瞳孔變化在目標呈現後約 600–1,100 ms 的時間窗觀察到。 | 生理訊號可用作補充，須保留時間延遲。 [Experiment 2，Figs. 4–5](https://doi.org/10.1371/journal.pone.0146194) · [瞳孔資料](https://doi.org/10.7910/DVN/QGEAL1) |
| **R27** Timkey et al.（2026） | 英語暫時性句法歧義在消歧詞及下一詞造成延滯。多種 LM surprisal 相對接近前向閱讀效應，回讀時間的增幅則大得多。 | 作詞級位置、延滯及回讀的控制測試。 [Figs. 1、3](https://pmc.ncbi.nlm.nih.gov/articles/PMC13462876/#fig03) · [資料與程式](https://osf.io/b6rqh/) |
| **R28** Hirotani, Frazier & Rayner（2006） | 英語句子加入標示子句末尾的逗號。實驗 1 局部首遍較慢，但整句閱讀時間縮短。 | 檢驗只改標點時，理解分數是否被局部閱讀行為誤導。 [Experiment 1](https://pure.mpg.de/pubman/faces/ViewItemFullPage.jsp?itemId=item_725746_4) |
| **R29** Xue et al.（2026） | 簡中眼動研究把同一目標詞放在句末或句中。多項指標呈現不同模式，總注視時間在目標句末較短。 | 與英語邊界效應分開按語言及眼動指標測試。 [Fig. 1、Table 4](https://doi.org/10.1177/17470218251347264) · [資料與程式](https://osf.io/xf2q9/) |
| **R30** Smith & Levy（2013） | 英語眼動與自定步速語料支持詞的條件機率取負對數與閱讀時間相關。效果可出現在後續詞，兩種方法的延滯模式不同。 | Surprisal 是控制變項。逐詞位置評估保留當前詞及後續區域。 [Figs. 3–4](https://www.mit.edu/~rplevy/papers/smith-levy-2013-cognition.pdf) |

### 心理語言學與篇章語言學

| 研究 | 設計與觀測 | 對 CogLLM 的用途 |
| --- | --- | --- |
| **R31** Christianson, Hollingworth, Halliwell & Ferreira（2001） | 英語成人閱讀暫時性句法歧義句。實驗 2 的從句角色題錯誤較多，主句角色題多數答對，顯示錯誤解讀可殘留。 | 同一句分開測每個語意關係，記錄歧義來源、消歧區段及讀者誤解的角色。 [Experiment 2，Table 5](https://ferreiralab.faculty.ucdavis.edu/wp-content/uploads/sites/222/2015/05/Christianson-et-al.-2001_LingeringInterpretationsGP_Cog-Psych.pdf) |
| **R32** Camblin, Gordon & Swaab（2007） | 在英語段落操弄目標詞的廣域篇章一致性和局部詞彙聯想。即使句子局部可解讀，較遠篇章的一致性仍影響眼動及 ERP。 | 固定局部詞義的合理性，改動上游語境檢驗篇章整合。理解正確率需另驗證。 [Experiments 1–5](https://pmc.ncbi.nlm.nih.gov/articles/PMC1766924/) |
| **R33** Claus & Kelter（2006） | 透過倒敘改變事件在文中的出現順序。讀者對較早故事時間的事件較難提取，即使剛在倒敘提過。實驗 3 使用辨認測驗。 | 將事件在文字中的距離和故事時間距離分開，依原研究的辨認端點評估。 [Experiments 1–4](https://research.uni-hannover.de/en/publications/comprehending-narratives-containing-flashbacks-evidence-for-tempo/) |
| **R34** Cozijn, Noordman & Vonk（2011） | 荷語讀者閱讀有／無因果連接詞的句對。明示因果使緊接連詞的區段較快、子句末端較慢，並使實驗 2 的推論陳述驗證較快。 | 對比線上整合和讀後推論的測量位置及反應時間。 [Experiments 1–2，原文](https://research.tilburguniversity.edu/files/1442185/cozijn_noordman_vonk2011.pdf) |
| **R35** Kleijn, Pander Maat & Sanders（2019） | 794 名青少年讀 20 篇荷語真實文本的連詞操弄版本。局部克漏字表現依關係類型而變，因果和對比明示時較好，添加關係明示時較差。全文合併未觀察到效應。 | 薄弱連貫與冗餘提示可能有不同方向。局部克漏字和全文表現分開檢驗。 [Results，作者機構全文](https://research-portal.uu.nl/en/publications/comprehension-effects-of-connectives-across-texts-readers-and-coh/) |
| **R36** Zhan, Levy & Kehler（2020） | 華語篇章續寫實驗操弄主動、把字和被字結構，以及隱含因果。句法形式和篇章預期改變代詞可能指向對象的分布。 | 三個發布 CSV 分別有 1,651、2,334、2,741 筆逐人續寫，含原文、操弄條件和再提及對象。發布表的對象編碼均為兩個候選之一，可評估留存反應的指涉分布。 [Experiments 1–3](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0237012) · [實驗 1 CSV](https://osf.io/download/8zk57/) · [實驗 2 CSV](https://osf.io/download/z48qy/) · [實驗 3 CSV](https://osf.io/download/yajhg/) |
| **R37** 伍麗梅、莫雷（2010） | 中文說明文三個自定步速實驗。在前提可提取時，明示與隱含前提均出現結論句推論訊號。插入兩個句子後，距離削弱此效應。 | 以原研究的前提距離操弄與結論位置重現自定步速閱讀效應。讀後解題準確率目前缺相應觀測。 [實驗 1–3，原刊](https://journal.psych.ac.cn/xlxb/CN/abstract/abstract210.shtml) |
| **R38** Scholman, Yan, Yung & Demberg（2026） | 兩項華語自定步速閱讀實驗操弄因果與讓步的成對連接標記。先出現的標記對第二標記的閱讀時間沒有可重複的促進效果。 | 以陰性對照檢查模型是否對連接標記一律給出負荷下降。研究主端點為閱讀時間。 [§4–6](https://link.springer.com/article/10.1007/s10936-026-10269-2) · [資料與程式](https://osf.io/frh6z) |
| **R39** Yu, Hsiao & Lin（2025） | 華語母語者閱讀含量詞、連詞及可分級形容詞的短對話，選出續句。語境對字面與語用解讀的影響，初步結果隨所測詞類而不同。 | 固定「有些」等片語，比較語境如何改變「有些但非全部」的推論分布。 [§3–4，原刊](https://www.benjamins.com/catalog/cld.25002.yu) |
| **R40** Wei et al.（2024） | 華語把字句中的句法與語意訊息交互作用，在自然眼動與自定步速閱讀得到不同模式。 | 華語局部訊號的測量方法對照。 [Experiments 1–2，Tables 3–4](https://doi.org/10.1017/langcog.2023.42) · [資料](https://osf.io/5vfwr/) |
| **R41** Bian, Yoon & Komachi（2026） | 成人閱讀英語醫學摘要或白話改寫。分別收集逐句理解自評、處理容易度和可查閱全文的選擇題作答。 | 可做句級監督，也可測自評與答錯風險的關聯。作答對應到證據句，這個對應本身未標註讀者在哪裡卡住。 [資料與方法](https://aclanthology.org/2026.readi-1.4/) |

### 未解資訊、後續線索與理解更新

| 研究 | 設計與觀測 | 對 CogLLM 的用途 |
| --- | --- | --- |
| **R44** Love, McKoon & Gerrig（2010） | 敘事中的角色若缺少與故事的關聯，該角色名字在辨認探針中反應較快，後續詞的辨認反應較慢，後續資訊的記憶也較弱。 | 分別測未解資訊的可提取性和它對後文處理的影響。 [Experiments 1–4，Tables 4–5](https://pmc.ncbi.nlm.nih.gov/articles/PMC2864939/) |
| **R45** Jones & Bergen（2024） | 英語短文操弄代詞是否有歧義，以及後文是否符合較常見的指涉解讀。實驗 1 收指涉選擇，實驗 2 在後續區段測到違反較常見解讀的閱讀變慢，並用明確指涉版本控制續句差異。 | 可對照前段的解讀與後段何時顯現困難。兩個實驗提供材料層級對照，後續區段的時間和指涉答案未在同一試次量測。 [Experiments 1–2](https://doi.org/10.1017/langcog.2024.2) · [材料、資料與程式](https://osf.io/v8rjm/) |
| **R46** Kendeou, Smith & O’Brien（2013） | 反駁後加入一句因果解釋，後續已無可測得的理解干擾，但仍觀察到舊資訊再活化。三句因果解釋及等長的限定性補述並有不同結果。 | 分開檢驗後續干擾是否下降，以及舊解讀在後文被再度喚起的程度。 [Experiments 1–2、6–7](https://experts.umn.edu/en/publications/updating-during-reading-comprehension-why-causality-matters/) |
| **R47** Pettijohn & Radvansky（2016） | 操弄敘事的事件轉換是否預告。實驗 3 未預告的轉換造成局部閱讀變慢，預告條件下未測到同樣的差異。先前內容的辨認反應在轉換後較慢，兩種預告條件均見此趨勢。 | 用預告操弄分開檢驗新情境帶來的局部處理延滯，以及舊資訊是否仍容易提取。 [Experiment 3，Table 5](https://memorylab.nd.edu/assets/259391/pettijohn_radvansky_2016_memory_cognition_.pdf) |

## 1980–1999 經典實證

| 研究 | 設計與觀測 | 對 CogLLM 的用途 |
| --- | --- | --- |
| **R42** Daneman & Carpenter（1980） | 建立 reading span，並探討它與事實提取、代詞指涉的關係。 | 提供工作記憶與理解個體差異的早期測法。個人分數須直接量測。 [原始研究的 ERIC 紀錄](https://eric.ed.gov/?id=EJ230538) |
| **R43** Yeung, Jin & Sweller（1998） | 五個閱讀實驗比較逐文解釋、集中列出的詞彙解釋等安排。文內整合的理解效應在五年級母語組與成人組方向相反，低與高能力的英語學習者也呈現不同模式。 | 以同一術語與兩種解釋排法做對照，分開測文本理解和詞彙學習。 [Experiments 1–5](https://pubmed.ncbi.nlm.nih.gov/9514686/) |
| **R01** McNamara, E. Kintsch, Songer & W. Kintsch（1996） | 英語青少年閱讀局部／全局連貫程度不同的心臟病說明文。實驗 2 的推論和解題表現呈現先備知識與連貫性互動。推論、解題、分類與文本回憶各自分析。 | 依題型、知識分層檢驗，並保留原文與改寫長度差。 [Experiment 2，Fig. 4、Appendix B](https://doi.org/10.1207/s1532690xci1401_1) |
| **R02** McNamara & Kintsch（1996） | 高／低知識讀者閱讀不同連貫性的歷史說明文。高知識讀者在低連貫版本上的關鍵詞分類表現較佳，理解結果依題型而變。 | 保留可能受益於主動推論的條件。 [兩個實驗，Results](https://doi.org/10.1080/01638539609544975) |
| **R03** Britton & Gülgöz（1991） | 用理解模型定位一篇教學文章中需要讀者自行推論的區域，合併修補這些區域。修訂增加大學生的自由回憶，另一研究的概念關係較接近專家。 | 直接引出局部修補實驗。論文的效應屬整套修補，單一區域的效應仍需測量。 [Experiments 1–2](https://doi.org/10.1037/0022-0663.83.3.329) |
| **R11** Chi, de Leeuw, Chiu & LaVancher（1994） | 八年級生讀人體循環說明文。有逐行自我解釋提示的組別，其知識前後測進步較多。另分析複雜題、隱含元件功能和心智模型。 | 讓人類說出預期的推論步驟，區分提示後表現和自行閱讀表現。 [Methods、Results](https://doi.org/10.1207/s15516709cog1803_3) |
| **R13** Recht & Leslie（1988） | 64 名中學生閱讀同一棒球半局敘述。以事先測得的棒球知識及閱讀能力分組。知識與動作重演、口頭複述、摘要及重要性分類均有關。 | 領域知識可以在固定文章下比較。此為預測分組差異，臨時教學的效果需另測。 [Methods、Results](https://www.yesataretelearningtrust.net/Portals/0/Effect-of-Prior-Knowledge-on-Good-and-Poor-Readers-Memory-of-Text.pdf) |
| **R15** McKeown, Beck, Sinatra & Loxterman（1992） | 學童閱讀原版或增進連貫的歷史材料。提高連貫性改善回憶及理解題表現。文章並比較具不同先備知識的讀者。 | 設計事件因果鏈的改寫測試，分開記錄材料改動和先備知識。 [完整作者研究稿，Results](https://files.eric.ed.gov/fulltext/ED410308.pdf) · [期刊版本](https://doi.org/10.2307/747784) |
| **R19** Harp & Mayer（1998） | 以閃電形成教材加入有趣但與學習重點無關的細節，測主旨回憶和遷移。細節可降低兩項表現，實驗 4 顯示位置影響結果。 | 對照增加解釋與等長增加旁枝資訊。教材含圖文，純文本遷移需另外量測。 [Experiments 1–4](https://doi.org/10.1037/0022-0663.90.3.414) |
| **R21** Albrecht & O’Brien（1993） | 敘事中的關鍵句與更早的角色描述有一致和不一致版本。實驗 1 在關鍵句觀察到不一致時閱讀較慢。 | 分別記錄觀測位置與造成不一致的來源句。 [Experiment 1](https://doi.org/10.1037/0278-7393.19.5.1061) |

## 理論、量測與載體

- **Sweller（1988）**以問題解決與基模習得說明有限處理資源在學習中的作用。這篇提供認知負荷理論的脈絡，文獻庫另以直接的閱讀實驗建立文本操弄。 [原文](https://doi.org/10.1207/s15516709cog1202_4)
- **Paas, Tuovinen, Tabbers & van Gerven（2003）**綜述心智努力評分、行為與生理量測，適合作為選擇量尺及解釋端點的依據。 [原文](https://doi.org/10.1207/S15326985EP3801_8)
- **DeLeeuw & Mayer（2008）**在電動馬達動畫教學中，發現次要任務反應時間、自評心智努力和難度評分對不同操弄的敏感性不同。使用此研究規劃量測，文本閱讀效度另驗證。 [Experiments 1–2，Table 6](https://doi.org/10.1037/0022-0663.100.1.223)
- **Delgado, Vargas, Ackerman & Salmerón（2018）**對 2000–2017 年的紙本與數位閱讀比較做後設分析，結果包括限時與文體之差異。此處用來界定閱讀載體與時間條件，結果端點是理解。 [作者全文，Abstract、Methods](https://www.uv.es/lasalgon/papers/Delgado%202018%20dont%20throw%20away%20your%20printed%20books.pdf)

## 可接入訓練與驗證的資料

| 資料 | 已公開記錄 | 可用標註與範圍 |
| --- | --- | --- |
| **D01 OneStop Eye Movements ＋ OneStopQA／STARC** | 英語新聞段落，真實讀者逐次作答、難度版本、閱讀目的及證據範圍。官方文件列出 `selected_answer`、`is_correct`、`onestopqa_question_id`、`question_preview`。 | 可做有作答的段落內風險基線。Critical span 是答題所需資訊，難點本身需獨立標註。原版和簡化版一起依來源文章分割。 [資料論文](https://www.nature.com/articles/s41597-025-06272-2) · [欄位](https://lacclab.github.io/OneStop-Eye-Movements/variables) · [STARC](https://aclanthology.org/2020.acl-main.507/) · [下載與授權](https://github.com/lacclab/OneStop-Eye-Movements) |
| **D02 Tibken & Tiffin-Richards** | 研究有 96 位德語讀者，發布分析檔涵蓋 94 位、1,870 筆試次。以一詞改動操弄矛盾，讀後回報有無矛盾，回報有時再從編號句中選位置。 | 矛盾條件的正確性同時要求有偵測到且選對句。發布檔只有合併的正確性，沒有單獨的偵測回報或所選句索引。公開材料提供示例篇章，重做所有文章的模型對照需完整材料。 [Methods、Data availability](https://link.springer.com/article/10.1007/s11409-025-09440-2) · [試次檔](https://osf.io/download/fcs25/?view_only=1cd488d47c254163bfc032fbbc7fa0cc) · [示例材料](https://osf.io/download/8qha3/?view_only=1cd488d47c254163bfc032fbbc7fa0cc) |
| **D03 Hwang et al. RECO（2026）** | 776 道由英語閱讀材料衍生的判斷題，專家標註證據句、證據範圍與推論類型。 | 可監督「給定一項陳述，哪些原句支持它，需何種推論」。此項評估帶有陳述輸入。公開集合用於示範或留出評估，依作者的研究用途授權使用。目前缺乏相應讀者答錯率。 [§3、§5](https://aclanthology.org/2026.acl-long.1241/) · [作者資料](https://github.com/SeonjeongHwang/ReCo) |
| **D04 QASA** | AI/ML 論文的讀者提出問題，專家給答案、引用段落及解釋。 | 可練習核對跨段證據及設計 how／why 題。適用範圍是科學論文的題目與證據。 [§3、Fig. 1](https://proceedings.mlr.press/v202/lee23n.html) · [資料](https://github.com/lgresearch/QASA) |
| **D05 WikiDomains** | 13 個學科、約 22,000 筆英文短定義，每筆搭配困難概念候選，另有人類對改寫的評估。 | 可提供概念片語弱標註及補充解釋素材。訓練難度分數需同一條件下的讀者反應。 [§3–5](https://aclanthology.org/2024.emnlp-main.357/) · [資料與授權](https://github.com/google-deepmind/wikidomains) |
| **D06 ExpMRC** | 中英文閱讀理解題與人類標出的證據片段，包含抽取式與選擇題。 | 中文和英文的證據範圍輔助任務。題目是輸入，標註描述證據位置。 [§2–3](https://pmc.ncbi.nlm.nih.gov/articles/PMC9048090/) · [資料與授權](https://ymcui.com/expmrc/) |
| **D07 PoTeC** | 德語物理與生物教材的逐詞眼動、主修、背景題與理解題。背景題在讀後施測。 | 用主修做閱讀前的群體條件，讀後背景題依讀後測量處理。眼動可作輔助控制。理解題未密集對齊到詞級難點。 [Experiment procedure、Table 12](https://link.springer.com/article/10.3758/s13428-024-02536-8) · [資料](https://osf.io/dn5hp/) |
| **D08 繁中 MECO 延伸語料** | 60 名香港讀者、12 篇繁中百科式短文，分橫排與直排並有篇章理解題。 | 繁中一般閱讀對照。局部專業推論與片語難點需要額外標註。 [Methods、Data records](https://www.nature.com/articles/s41597-026-06989-8) |
| **D09 PLABA-EVAL** | 39 對英語摘要及白話改寫、609 句、1,826 筆納入分析的逐句雙量尺評分。原始資料另含疑難類型、自由回饋、全文評分、開卷作答及作答容易度。 | 已用自評訓練句級頭。逐句類型可做另一個有明確負例的端點。原任務給全文，兩種分數的觀測粒度都是句。 [論文](https://aclanthology.org/2026.readi-1.4/) · [CC BY 4.0 資料版本](https://github.com/SDS-NLP/PLABA-EVAL/tree/153082e7d58b833d5e2f481b92fdcf9248c60d11) |
| **D10 INQUISITIVE** | 1,500 篇英語新聞的前五句逐句揭露。讀者讀完當前句，圈出原文片語並提出問題。公開問題檔有 19,816 筆，附焦點句與圈選位置。 | 有句末停點和片語來源。公開的問題檔缺少完整的「沒疑問」分母及同一疑問的後續狀態。全文依 WSJ、AP、Newsela 等來源另取得。 [論文 §3.1–3.2](https://aclanthology.org/2020.emnlp-main.530.pdf#page=3) · [公開問題檔](https://github.com/wjko2/INQUISITIVE/blob/master/questions.txt) · [文章取得方式](https://github.com/wjko2/INQUISITIVE#article-sources) |
| **D11 TED-Q** | 111 位真人讀六篇英文演講逐字稿的片段。原始逐次檔有 2,412 個讀者提出的疑問，以及 3,669 筆對先前疑問的後續評分。提問與有答案時均可標出關鍵片語。 | 用讀者和疑問 ID 串接最長兩次後續停點。每兩句一停，量尺是疑問被回答的程度。可作區間更新的實驗先例。TED 現行逐字稿使用條款將 AI/ML 開發及評估列為需另行授權的用途。 [論文 §3–4](https://aclanthology.org/2020.lrec-1.141.pdf#page=3) · [原始 CSV 與欄位](https://github.com/amore-upf/ted-q) · [TED 條款](https://www.ted.com/about/our-organization/our-policies-terms/ted-com-terms-of-use) |
| **D12 Reading between the Lines** | 50 位讀者、57 篇英文書評。真人圈出自己不理解或不熟悉的片語，另有整篇熟悉度、理解自評、讀者背景、是否讀過書或作者。 | 可估全文可見後的片語標記及讀者背景差異。發表檔的片語類型由 GPT-4o 標，片語字元偏移須從文字重建。原始書評與標註的訓練使用條件待確認。 [論文 §3.2–3.3](https://aclanthology.org/2025.naacl-long.409.pdf#page=3) · [CSV 和欄位](https://github.com/sougata-ub/reading_between_lines) |
| **D13 Im & Färber 科學摘要** | 英語跨領域讀者讀原摘要與改寫，自行勾選困難句、圈出複雜詞或片語，並比較兩版理解容易度。論文報告每個比較維度各 92 筆判斷。 | 可做全文可見後的句／片語定位及改寫偏好。原始備份可用 passage ID 串接文本，但圈選欄位是純文字，含多段與留言。正式分析的 92 筆映射及 SciSummNet 原文條件仍待確認。 [預印本 §3.3、§4.1](https://arxiv.org/html/2607.25630v1#S3.SS3) · [作者備份及授權說明](https://github.com/faerber-lab/scientific-text-simplification-corpus) |
| **D14 CompLex** | 真人在英文句子中用五點量尺評指定詞或多詞表達的難度。官方版發布彙整分數。 | 可訓練指定詞／片語的主觀詞彙難度。位置由資料先指定，觀測單位是指定詞，整句其他詞未取得同等評分。 [SemEval 方法](https://aclanthology.org/2021.semeval-1.1/) · [CC BY 4.0 資料](https://github.com/MMU-TDMLab/CompLex) |

### Hugging Face 來源查核

2026 年 11 月 20 日以免登入的 Hugging Face Hub API 核對首批候選的可見性與檔案。資料入口的用途按所附端點決定，並核對原文與標註的來源。列為作者來源的項目在本輪未找到含相同真人欄位的公開 Hub 版。

| 項目 | 可用的入口與用途 |
| --- | --- |
| **Qwen3.5-2B** | 官方 model repository 可公開下載。權重、設定與 tokenizer 固定在已查核的 `15852e8c16360a2fea060d615a32b45270f8a8fc`。 [官方固定版本](https://huggingface.co/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc) |
| **PLABA-EVAL（D09）** | 固定使用作者 GitHub 的 CC BY 4.0 版本。該發布包含逐句兩量尺、自由回饋、疑難類型及逐題作答所需的表格。 [作者固定版本](https://github.com/SDS-NLP/PLABA-EVAL/tree/153082e7d58b833d5e2f481b92fdcf9248c60d11) |
| **華語指涉續寫（R36）** | 使用作者 OSF 的 CC BY 4.0 發布。已查驗三個實驗的逐次檔。 [原始資料與刺激](https://osf.io/t87my/) |
| **OneStopQA 與真人作答（D01）** | 作者連結的 `malmaud/onestop_qa` 可從 Hub 直接取得文章、題目、答案、證據及干擾區段。真人作答從作者 GitHub 的原始行為實驗或作者發布的 OneStop Eye Movements 取用，並按各自的研究條件評分。 [Hub 版題目與證據](https://huggingface.co/datasets/malmaud/onestop_qa) · [作者原始行為實驗](https://github.com/berzak/onestop-qa/tree/master/human_experiments) · [OneStop Eye Movements 原始資料](https://osf.io/2prdq/) |
| **INQUISITIVE（D10）** | `UT-CompLing/inquisitive_qg` 可公開查看，主分支有資料卡與載入腳本，完整前綴仍須依作者的文章取得方式。在所需文本及其使用條件核對完成後，才納入訓練或驗證。 [Hub 檔案](https://huggingface.co/datasets/UT-CompLing/inquisitive_qg/tree/11b9160bec51fb01e2f1999f0de1c399aa81567a) · [作者的全文取得說明](https://github.com/wjko2/INQUISITIVE#article-sources) |

### 首輪訓練頭的資料盤點

- **句級自評與疑難類型。** 依 D09 固定版本的 `annotations.csv`，1,826 筆納入評分分析的讀者×句子中，858 筆用結構化疑難勾選介面，涵蓋 286 句。452 筆至少勾選一項，406 筆明確回報無問題。術語不清有 335 筆，資訊太密 102 筆，措辭晦澀或冗長 85 筆，重點不清 54 筆，指涉不明 20 筆，其他 12 筆，類型可重複。其餘 968 筆採自由文字介面，其中 449 筆有留言。訓練類型頭時只在相應欄位的觀測上計算損失。發布檔標為納入的 935 筆開卷作答有正誤及 1–5 級作答容易度和信心。歷史基線以作答表索引連結 932 筆作答，其中 213 筆的句子落在題目表證據範圍外。Stage 2 另從題目表建立證據清單。主題熟悉度與學術閱讀背景是自陳資料，研究未設閱讀前的知識測驗。 [原論文 §3.2、Fig. 1](https://aclanthology.org/2026.readi-1.4.pdf#page=3) · [句級資料與欄位](https://github.com/SDS-NLP/PLABA-EVAL/tree/153082e7d58b833d5e2f481b92fdcf9248c60d11#annotationscsv) · [Stage 2 證據句訂正](stage-2-data.md#plaba-eval-證據句訂正)
- **全文可見後的真人圈選。** D12 發布 CSV 有 933 列。以讀者×書評去重為 611 筆試次，其中 312 筆沒有標區段，299 筆有標區段，共 621 個區段列。611 個區段列在全文內只有一次符合的子字串，10 個有多個候選位置。未圈選是該介面的標記結果。要知道是否在第一遍讀到某個詞時卡住，需額外記錄當時的停點。論文報告品管後 668 筆，這裡的數量以發布檔的試次鍵計。 [公開 CSV](https://github.com/sougata-ub/reading_between_lines/blob/main/user_study_data.csv) · [論文 §3.3](https://aclanthology.org/2025.naacl-long.409.pdf#page=4)
- **同一疑問的區間變化。** D11 的 `worker`、`prior_question`、`chunk_number` 能對應後續評分。原始檔有 2,562 筆未回答（1–2 分）與 1,107 筆至少部分被回答（3–5 分）。保留每個停點的分數，未在追蹤期間獲答的疑問以觀測截止處理。被回答的片語有原文偏移，讀者何時察覺答案的直接觀測精度為兩句。 [原始資料與欄位](https://github.com/amore-upf/ted-q) · [論文 §3–4](https://aclanthology.org/2020.lrec-1.141.pdf#page=3)
- **理解結果與讀者差異。** D01 的真人作答及閱讀任務可查驗篇章理解是否隨改寫或閱讀目標變動。D07 的主修與修業階段可作閱讀前背景，理解與領域知識另有各自的題目。華語的 R36 續寫可查驗同一語境下的指涉解讀分布。作答錨點是題目所依據的資訊，真人反應的發生時點另外量測。 [OneStop 欄位](https://lacclab.github.io/OneStop-Eye-Movements/variables) · [PoTeC 實驗程序](https://pmc.ncbi.nlm.nih.gov/articles/PMC12208991/) · [華語材料與續寫實驗](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0237012)

現有可用資料支持句級自評、句級疑難類型、符合素材使用條件的圈選位置，以及讀後理解結果。文獻也有兩句一停的疑問更新先例。要把「所指區段留在前句、困難卻出現在下一句開頭」當作逐 token 的實證結論，需要既有資料在關鍵詞前後都有對應的反應觀測。已核對資料若只有句末回報、全文圈選或較粗的追蹤區間，就依各自的觀測粒度驗證，逐 token 觸發與未觀測的轉移列為模型假說。 [數學規格與觀測似然](mathematical-formulation.md#4-人類觀測與聯合訓練)

## 從論文讀到的效應如何使用

- **有原始觀測時**，保留文章、版本、受試者或群組、閱讀指令、題型、目標區域、端點、樣本數和資料來源。原始研究與重分析共享同一分割群組。R07 和 R08 是同資料的例子。
- **只有發表的統計量時**，登錄效應方向、估計值、單位與報告的不確定性。這些摘要用來設定重現測試或檢查效應的合理範圍。推論題答對率的條件差由受試者作答估計，不能據此把文章中每個詞填上同一個強度。
- **只有素材與實驗對照時**，可以重建材料、操弄及模型刺激。缺少對應的既有人類反應時，結果可用於工程查核或提出假說。讀者群、語言或題型的更換都要在結果中標明。
- **結果未達顯著時**，保留為不確定或待重現。若要主張兩個條件近似，先設定具有實質意義的等效區間，再以足夠精度檢驗。

[Qwen3.5-2B 訓練與驗證方案](training-protocol.md)
