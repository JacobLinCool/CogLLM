# Stage 1 訓練與驗證訊號選擇

本文件記錄已核對的既有真人觀測及其使用範圍。初版保留不同端點的原量尺與可見條件。閱讀時間、理解作答、理解自評和疑問圈選分別統計。文章與讀者背景仍是 App 的一般輸入。問答端點只在研究對照中使用，按各實驗設定提供題目及選項。 [訓練規格](../training-protocol.md) · [數學規格](../mathematical-formulation.md#4-人類觀測與聯合訓練)

## 優先接入的資料

| 來源 | 觀測與版本查核 | 首輪用途與切分 |
| --- | --- | --- |
| **D09／R41 PLABA-EVAL** | 英語生醫摘要，39 個來源配對、609 句、1,826 筆可看全文的句級雙量尺評分。858 筆另有結構化疑難回報，其中 452 筆勾至少一項，406 筆明確無疑難。公開檔標記 935 筆開卷作答為納入分析。版本與資料使用條件鎖於作者 CC BY 4.0 版本。 | **開發訓練。** 句級容易度、理解自評及已量測類型。結構化介面有「確定沒有疑難」欄，自由文字空白依原介面處理。原文與白話版同組。來源文章五折有 9 個讀者跨折，因此讀者泛化另按共用讀者連通群分割。 [E01 與固定版本](../stage-1-observations.md#e01-plaba-eval-白話改寫與逐句理解自評) · [資料盤點](../reading-evidence.md#首輪訓練頭的資料盤點) |
| **R36 Zhan et al.** | 華語續寫有原始提示、條件、讀者、選擇對象。E02 已依作者分母及讀者等權重算自由提示的主語再提及率。OSF 專案標示 CC BY 4.0。 | **開發診斷及輔助監督。** 對照同續寫任務的指涉選擇，保留兩候選原文區段。原文、提示變體及同來源實驗合併。 [E02 及原檔雜湊](../stage-1-observations.md#e02-zhan-et-al-自由提示下的再提及對象) |
| **D01 OneStopQA／OneStop Eye Movements** | HF `malmaud/onestop_qa` 固定版本 `1df9aea8c9532045bc55c001d7c588372de2d723` 為 CC BY-SA 4.0 材料、題目、選項及 STARC 片段。作者同系的 Prolific 開卷 QA 表已讀取，1,940 筆屬 OneStop，另外 1,940 筆為 RACE。360 人的 OneStop Eye Movements 為另一批讀者，普通閱讀 OSF 檔記有 `selected_answer`、`is_correct` 與題目 ID，已確認發布位置及使用條件，尚未解析其大型檔。 | **優先接入真實篇章理解對照。** 問題與正文同時可見的 Prolific 和讀完正文才見題的普通閱讀各用獨立指令設定。答案證據或干擾證據可做題目給定後的區段檢查。來源文章、改寫、不同閱讀任務及兩個資料發布同家族。 [固定 HF 版](https://huggingface.co/datasets/malmaud/onestop_qa/tree/1df9aea8c9532045bc55c001d7c588372de2d723) · [作者試次說明](https://github.com/berzak/onestop-qa/tree/7a058755e1ff0d3f835a67109c669c7e294d4239/human_experiments) · [正式欄位](https://lacclab.github.io/OneStop-Eye-Movements/variables) · [普通閱讀資料](https://osf.io/zn9sq/) · [授權](https://github.com/lacclab/OneStop-Eye-Movements#license) |
| **D08 繁中雙向 MECO** | 60 名香港學生、12 篇繁中文章。讀者讀完每篇回答四個簡易是非題。作者 RDA 的 2,880 筆讀者×文章×題目皆有正誤，其中橫排答對 1,171／1,440，直排答對 1,147／1,440。文章 Data Records 指定整份資料為 CC BY 4.0，OSF 專案機讀欄位未填。 | **繁中篇章理解診斷候選。** 首輪限同一呈現方式，方向對照需有模型輸入對應。橫排、直排及可辨認的同一 MECO 原文／改編合組。 [原文 Methods、Data Records](https://pmc.ncbi.nlm.nih.gov/articles/PMC13096406/) · [作答 RDA](https://osf.io/download/3sfc7/) · [資料專案](https://osf.io/jyn8s/) |
| **R38 Scholman et al.** | 華語有／無成對連接詞的自定步速閱讀，24 組同源材料及兩個實驗。第一實驗已核對 5,428 個發布區段列、兩項實驗各有 96 列版本素材，OSF 標示 CC BY 4.0。 | **選用的閱讀時間輔助對照。** 依區段看差異並保留方向與不確定性。第二標記的合併交互作用接近零，估計區間跨零，因此不設硬性零值。在把係數方向寫入程式前核對作者編碼。 [R38 詳細紀錄](language-and-updating.md) · [原研究 Experiment 2，Exploratory: Pooled analysis](https://link.springer.com/article/10.1007/s10936-026-10269-2) |

### 可重算的 D01 開卷作答觀察

以作者固定版本的 `human_experiments/prolific/qa.tsv` 篩選 `source == 'Onestop'` 及 `trial_type == 'reading_comprehension'`，`answer_response == '0'` 代表選 A。**1,567／1,940** 道作答正確（80.77%）。進階版 774／970（79.79%），基礎版 793／970（81.75%），觀測差約 +1.96 個百分點。這是等試次權重的描述值。若要估計版本效應的不確定性，依讀者及原始題目／文章的分派保留群組結構。原試次檔 SHA-256 為 `bc8af0a78819b918eb0ce97935bf8fcd82eeec2d9f99111762034817c1e56e0d`。 [試次 CSV/TSV 入口與答案映射](https://github.com/berzak/onestop-qa/tree/7a058755e1ff0d3f835a67109c669c7e294d4239/human_experiments) · [STARC 原文 §2](https://aclanthology.org/2020.acl-main.507/)

## 有關定位、背景和更新的資料

| 來源 | 本次查核 | 可使用的觀測與缺項 |
| --- | --- | --- |
| **D10 INQUISITIVE** | 作者固定 `9c364b689a3c134cc5a0bc886f0ce7ce190e6111` 的 `questions.txt` 為七欄逐行 TSV。依作者 loader 按 tab 逐行讀取並保留內嵌引號，得到 19,816 個問題，涵蓋 1,500 個文章 ID 和 7,099 個有問題的文章×句子停點，限每篇前五句。 | **句末、條件式片語定位候選。** 發布檔只留正例，缺讀者 ID、無疑問停點及同一疑問的追蹤。指令要求定位當前句的片語，因此無法訓練跨句指向。依字面空格切分及零起點半開偏移，19,814 筆可精確取回非空片語。另有一筆空片語及一筆偏移與片語不一致，位置需要遮罩。完整前綴各來源需另取得使用許可，HF 固定版只有 loader 和說明，未列授權。 [原論文 §§3.1–3.3、Fig. 2](https://aclanthology.org/2020.emnlp-main.530.pdf#page=3) · [作者固定版與取件方式](https://github.com/wjko2/INQUISITIVE/tree/9c364b689a3c134cc5a0bc886f0ce7ce190e6111#article-sources) · [HF 固定版](https://huggingface.co/datasets/UT-CompLing/inquisitive_qg/tree/11b9160bec51fb01e2f1999f0de1c399aa81567a) |
| **D12 Reading between the Lines** | 固定作者 CSV `ab527b45fa96c434022e90a7058ad1e698197d82` 的 933 列合為 611 組讀者×書評，299 組圈選，312 組未圈選。621 個圈選區段有 611 個可唯一字面定位，8 個出現兩次、2 個三次。原文 §3.3 報品管後 668 個回應，兩個分母的映射未提供。 | **全文可見的個人圈選候選。** 沒有線上停點。若作 App 標記，用真人區段，發布類型由 GPT-4o 編碼。作者發布庫無資料授權，原 Goodreads 來源說明限學術用途及禁止再分發、商業使用。先保留設計與發布分母查核。 [固定作者資料](https://github.com/sougata-ub/reading_between_lines/tree/ab527b45fa96c434022e90a7058ad1e698197d82) · [論文 §§3.2–3.3](https://aclanthology.org/2025.naacl-long.409.pdf#page=4) · [上游資料使用條件](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home) |
| **D13 Im & Färber** | 作者固定版本 `9b116c2b2c398c369a681116a1480e87a3072c8b` 的 Phase 1 JSON 確有 92 筆評估，17 個讀者 ID、83 個篇章 ID。檔案只保存顯示位置 Passage 1／2 或相同，對理解選擇的各類筆數為 39／45／8。論文 Fig. 5 依版本報告白話版較好為 73／92，未做推論統計。SQL 讓 92 筆都能找到候選文章對，已查欄位未提供逐次的呈現順序。片語欄含自由文字，也未見困難句的索引。 | **改寫偏好彙總及介面設計。** 原文與簡化版的逐次偏好仍無法重算。文章對已經接起，圈選的片語位置仍需逐項重建，並須另外確認 SciSummNet 上游條件。 [原論文 §§3.3、4.1、Fig. 5](https://arxiv.org/html/2607.25630v1#S4.SS1) · [固定作者資料與授權說明](https://github.com/faerber-lab/scientific-text-simplification-corpus/tree/9b116c2b2c398c369a681116a1480e87a3072c8b) |
| **D11 TED-Q** | 2,412 個引導式回報的疑問、3,669 筆回訪。讀者、篇段和問題鍵能串起同一疑問。每兩句一停，觀測窗最多兩個後續停點。 | **更新設計依據。** 在申明觀測窗後可定義何時至少部分解答。後續未測時以截止處理。TED 文本作模型開發及評估需另有授權。 [E05 重算與授權](../stage-1-observations.md#e05-ted-q-同一疑問後續有沒有被解答) |
| **D07 PoTeC** | 作者發布的 `participant_response_accuracy.tsv` 固定版本 `2742221e70cb9d2b38499bd080e079e6c0c1f5a3` 有 75 名讀者×12 篇共 900 列。888 列填有三道文本理解及三道領域題，共 2,664 道可用理解題。學科匹配的研究生 636／831 題正確（76.53%），其他讀者×材料為 1,089／1,833（59.41%）。這是合併試次的描述差，涉及多個群體。領域題在讀後量測。 | **讀者條件設計與彙總觀測。** 可預填主修及修業階段。出版社取得教科書原文作發布刺激的許可，原文另有版權，模型重用條件仍待核對。群體差同時反映領域及修業階段。試次表 SHA-256 為 `e95ef76499da606d43ec20cc9971368e0c54918f7ae7f4358d399bb6f5aefe13`。 [目標材料與題目方法、Table 16](https://link.springer.com/article/10.3758/s13428-024-02536-8) · [原文資料聲明](https://pmc.ncbi.nlm.nih.gov/articles/PMC12208991/) · [固定發布檔](https://github.com/DiLi-Lab/PoTeC/blob/2742221e70cb9d2b38499bd080e079e6c0c1f5a3/participants/participant_response_accuracy.tsv) |

## 其他輔助來源

| 來源 | 判定 |
| --- | --- |
| **D02 Tibken & Tiffin-Richards** | 有 1,870 筆發表分析的偵測＋找對句合併正確性，公開表缺所選句索引，完整刺激待確認。與 R16 的刺激有同一任務來源，重疊需再比文字。 [R09／R16](measurement-and-boundaries.md) |
| **D03 RECO、D04 QASA、D06 ExpMRC** | 專家證據及推論資料。可另建給定題目時的證據段落對照。各資料的標記端點、來源文件和使用條件獨立保存，閱讀進程及一般讀者的反應強度需要其他觀測。 [RECO §§3、5](https://aclanthology.org/2026.acl-long.1241/) · [QASA §3、Fig. 1](https://proceedings.mlr.press/v202/lee23n.html) · [ExpMRC §§2–3](https://pmc.ncbi.nlm.nih.gov/articles/PMC9048090/) |
| **D05 WikiDomains、D14 CompLex** | 前者有挑出待解釋概念的資料，以及經人評估的特定改寫。後者有英語正文中指定詞及片語的 1–5 分彙整難度，官方發布附 CC BY 4.0。可提供概念和詞彙輔助端點。評分目標先指定，未被選的詞沒有同樣的觀測。 [WikiDomains §§3–5](https://aclanthology.org/2024.emnlp-main.357/) · [CompLex 方法](https://aclanthology.org/2021.semeval-1.1/) · [CompLex 固定版本](https://github.com/MMU-TDMLab/CompLex/tree/4b31c2bd34da4c1265f9db3d727d95ee23099f14) |
| **de Varda、Marelli 與 Amenta 的 Aligned 資料** | 固定發布表有 205 句、1,726 個目標詞，全部可按原始鍵對上材料表。每句的 1 到 14 詞前綴可依所見目標詞逐步接回，205 條鏈均逐字一致。每目標有真人的 1–5 可預期性評分均值及平滑後 cloze 機率，可檢驗單次前向中 logits 給實際後續詞的機率。這兩個端點對應預期或預期違反，讀者的特定疑問及解除時間需另外觀測。作者釋出的評分與 cloze 資料採 CC BY 4.0，原句及 Frank 等人的量測沿用上游條件。 [原研究](https://doi.org/10.3758/s13428-023-02261-8) · [固定版本與使用條件](https://github.com/Andrea-de-Varda/prediction-resource/tree/47b7f923a27111da428d3802d9ab52c6db835ba8#license) |

### Stage 3 並行查核的閱讀中回報資料

- **ROAMM** 有 44 位讀者閱讀五篇英語材料。讀者察覺走神時按鍵，再回溯圈選開始與結束詞。公開頁面表計得 2,200 次頁面暴露，1,001 次有回報。原始按鍵停點、回溯範圍及讀後理解可個別使用。讀者圈選的開始位置受頁面邊界限制，無回報只表示沒有按鍵回報。可評估自覺注意力中斷或用來核對「停點＋範圍」的資料結構。語意困惑與跨句指涉需要相應的讀者回報。OpenNeuro 發布標為 CC0，改編文本另有上游來源條件。 [ICML 2026 資料論文](https://proceedings.mlr.press/v306/sun26ag.html) · [方法與補充](https://arxiv.org/html/2611.14814v1#S2.SS1) · [OpenNeuro 1.3.0](https://openneuro.org/datasets/ds007629/versions/1.3.0) · [公開頁面表](https://s3.amazonaws.com/openneuro.org/ds007629/derivatives/behavioral/roamm_page_level_reading_data.csv)
- **Cambridge Dictionary Look-Up** 記錄第二語言讀者查閱哪些詞，可檢驗有讀者背景時的查詞行為。研究端點為實際查詞，閱讀中的所有理解疑問需另有觀測。正式發布需申請使用，資料重散布及衍生成果發布有額外條件。本輪未接入模型。 [CoNLL 2025 方法](https://aclanthology.org/2025.conll-1.32/) · [資料與使用條件](https://researchdatasets.cambridge.org/datasets/cambridge-dictionary-look-up-dataset)

R07／R08 共用同一個理解作答家族。R09／R16 先按可能重疊處理，R02／R03 也先視為可能同源的歷史說明文。原始文本、改寫、同家族重分析及答案衍生標註沿同一分割。D01 與 D08 可驗證各自的理解端點，正式跨研究留出仍需指定相同的端點與可見規則。已用於設計或開發的輸出留有使用紀錄。各家族與最終評分規則尚未封存。 [方法與界線](measurement-and-boundaries.md) · [跨研究驗證規格](../mathematical-formulation.md#真人效度)

本輪未找到同時涵蓋**逐詞停點、同一讀者的疑問回報、所指的任意已讀片語，以及後續同一疑問的更新**且有可核對重用條件的資料。App 第一版因此以句級自評和標記做第一個可驗證版面。點選所指上文及逐步解除的互動依實際可校準的觀測粒度接入。
