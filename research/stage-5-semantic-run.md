# Stage 5 語意定位、疑問更新與跨句銜接

本輪核對三個可分別觀測的量。**回報停點**是實驗要求讀者作答時已顯示的最後位置。**所指區段**是讀者指出與疑問有關的區段。**疑問更新**是在後續停點，由同一讀者對同一個疑問重新回報的狀態。只在資料有相應的觀測與分母時估計該端點。句末或兩句一停的資料給出回報時的可見前綴。句內首次察覺的 token 仍屬於未觀測區間。 [Stage 5 原設計](stage-5-scope.md) · [狀態轉移的區間量測，§5](mathematical-formulation.md)

## 問題指向與更新的資料核對

**INQUISITIVE 有 19,814 筆可精確對齊的圈選片語。** 作者版共發布 19,816 個已驗證的疑問，對應 7,099 個有提問的文章×句子停點。指令讓讀者在讀完當前句時提問並圈出當句片語。7,500 個排定停點中，有 401 個在發布的題目檔無對應資料列。讀者 ID、實際參與的分母及無問題回報未隨檔案發布，所以這 401 個停點的提問狀態保留為未知。作者要求申請完整文章。官方公開的題目列也缺少 463 個有回報的停點所需的至少一個上文句子。 [原研究 §§3.1–3.3](https://aclanthology.org/2020.emnlp-main.530.pdf#page=3) · [作者的文章取得說明](https://github.com/wjko2/INQUISITIVE/tree/9c364b689a3c134cc5a0bc886f0ce7ce190e6111#article-sources) · [公開列重算](results/stage5-semantic/inquisitive-audit.json)

DCQA 對同一批新聞文章另外發布 719 筆專家回答判定，其中 422 筆有可解析的答案句索引。這些位置可作為文本哪裡給出答案的標註。讀者是否在那裡察覺答案，須用讀者自己的後續回報驗證。719 筆都可用文章與問題文字找到 INQUISITIVE 同源紀錄，做留出時必須按文章家族合併。 [DCQA §6.1](https://aclanthology.org/2022.emnlp-main.806.pdf#page=7) · [同源對齊稽核](results/stage5-semantic/inquisitive-audit.json)

**TED-Q 支持有截止時間的疑問更新。** 任務在每個非末尾停點要求提出疑問，提問率的分母受此指令限制。每兩句一停，共有 2,412 個原疑問各獲一次後續評分。第一次有 910 個評為至少部分得到回答（1–5 分中的 3 分或以上）。其餘 1,502 個中，有 1,257 個被第二次評分，197 個在第二次達標。245 個只觀測到第一次，因篇段結束而截止。題目達到 3 分後即退出後續追蹤，因此「至少部分回答」之後是否完全解開需要另一筆觀測。40 個讀者的末停點紀錄有字元終點早於起點的問題，對應 66 次仍可按讀者與停點索引核對的序位評分。 [原研究 §§3–4](https://aclanthology.org/2020.lrec-1.141.pdf#page=3) · [逐次評分重算](results/stage5-semantic/tedq-audit.json)

1,257 個同次追蹤兩個疑問的停點中，有 **88 次只有較舊的疑問達到 3 分**，317 次只有較新的疑問達標，109 次兩個都達標，743 次兩個都未達標。若把同一停點的 3 分視為部分回答，88 次提供了對固定先解較新疑問規則的反例。另外有 109 個疑問從第一次的 2 分變成第二次的 1 分。評分問的是新顯示的片段回答了多少，這個量尺允許往下變動。每兩句一次的評分無法確定區間內哪個 token 產生更新。TED 的逐字稿使用條款將 AI/ML 開發及評估列為需另行授權，本輪用公開欄位重算描述統計。 [評分介面](https://github.com/amore-upf/ted-q/blob/10fb7e1a2b2d756f8ea190ee2ac223fecaec0f20/data_gathering/elicitation/chunk_includes/fragment_answer.html#L126-L139) · [逐字稿條款 §§6.1、6.3–6.4](https://www.ted.com/about/our-organization/our-policies-terms/ted-com-terms-of-use#h--copyright-issues-and-licenses) · [兩疑問的同次評分](results/stage5-semantic/tedq-audit.json)

衍生版 TED-QDB 公開 2,391 個疑問，保留提出時的原文終點，每題只有一個答案分數。QSalience 的 311 個人工判斷評的是整段剩餘正文能否回答問題。這兩個發布檔均沒有原提問者在各停點的評分序列。 [TED-QDB 固定版](https://github.com/LeidenNLP/ted-qdb/tree/281b6d213575b5c58c6d2dd2c0bee1010e85ca69) · [QSalience §5](https://aclanthology.org/2024.emnlp-main.1114.pdf#page=5)

**EvoQue 的非 TED 材料具有相近的追蹤設計。** 兩篇 DISCO‑SPICE 對話逐字稿收集 2,131 個提問，另有一篇練習故事的 222 個提問。讀者在後續停點回報同一疑問的回答程度，可以標出相關片語。作者發布的非 TED 項目是供人工查看的靜態視覺化。本次清查公開檔案未取得可串接每位讀者、同一疑問、後續評分與回答區段的逐筆資料，也未確認逐字稿及衍生標註的模型使用授權。 [原論文 §§3–4](https://pragmatics.ppls.ed.ac.uk/~hannah/papers/WesteraRohde.2019.pdf#page=5) · [公開視覺化](https://evoque-data.github.io/) · [上游語料](https://johnmkirk.co.uk/research-projects-SPICE-ireland)

逐一解析公開視覺化的三份非 TED 材料，找到了 1,347、784 和 222 筆疑問列。每列都有顯示用化名、問題與引用的觸發片語，部分片語是節錄。靜態頁面的 JavaScript 只切換顯示區段，公開檔未包含可按同一疑問接合的評分或回答列。展示案例與第一篇材料重複。 [固定發布檔案](https://github.com/evoque-data/evoque-data.github.io/tree/0b709e231a98e0f49dce20e7bab9d4624603e85e)

**Abel 的篇章順序效應可由公開資料重算。** 114 位讀者的釋出檔只有每輪閱讀的推論統計，缺少實驗逐段記錄。已用同一檔案重現交錯排序提高歸納推論的主效應及其與提問提示的交互。作者提供的兩版刺激是德文原文的英譯，可核對篇章排列，無法用來驗證個人疑問的原文位置。 [資料粒度與重算](stage6/abel-interleaving-audit.md)

## 其他可區別的觀測

Hippocorpus V3 有 80 個主題家族、240 篇故事、4,165 個句級停點和 33,134 筆既有讀者判斷。讀者逐句判斷有沒有新的事件及事件是否出乎預期，其中有 16,340 筆明確選擇「無新事件」。以逐筆票數可重現發布的全部句級比例。移除首句後，以原始票數重算的 3,925 個句分類與 Wang 等人表 1 有小幅差異，兩份分母及各類數量均保留在稽核。此端點可驗證事件邊界和意外感。語意疑問、所指片段及後續回答程度沒有相應欄位。Microsoft 曾宣布將 Hippocorpus 納入 CDLA-Permissive-2.0。本輪取得的 V3 壓縮檔未附新註記的單獨條款，保留為公開資料稽核。 [Hippocorpus V3 與方法](https://www.microsoft.com/en-us/download/details.aspx?id=105291) · [Sap 等人，收集事件標註](https://maartensap.com/pdfs/sap2022quantifying.pdf#page=7) · [Wang 等人，§3、表 1](https://aclanthology.org/2022.wnu-1.1.pdf#page=3) · [授權公告](https://www.linuxfoundation.org/press/press-release/enabling-easier-collaboration-on-open-data-for-ai-and-ml-with-cdla-permissive-2-0) · [發布檔重算](results/stage5-semantic/hippocorpus-audit.json)

補查發現，原論文的預期性分析限於重大事件，發布表也將所有次要事件列為符合預期。作者預印本進一步指出，80 個主題中有 60 個在表單上明確提問預期性，公開 CSV 缺少可辨識這 60 個主題的欄位。可直接重算的「有無新事件」與「重大事件中的意外比例」因此有不同分母。 [正式版方法](https://pmc.ncbi.nlm.nih.gov/articles/PMC9659415/#s2) · [預印本附錄 A.2](https://arxiv.org/pdf/2201.02662#page=11)

Tibken 的研究在全文讀完後由讀者口述含矛盾的句子編號。先前核對的資料只有合併的正確碼。這輪以公開網址查詢兩個 OSF 專案得到 HTTP 401。自動審查拒絕用網址裡的 view-only 令牌存取，原因是專案可能含非公開資料。現有公開量測保留在資料盤點中。 [Tibken 等人，Methods](https://link.springer.com/article/10.1007/s11409-025-09440-2) · [資料盤點 R09、R16](stage-1/measurement-and-boundaries.md)

**口述思考研究的公開表格停在群體或篇章指標。** Christhilf 等人的表格有 73 位讀者及各人的理解監控總分，作者說明隱私考量下未公開完整的回報內容。McNamara 等人的 RSAT 公開兩表分別有 495 個人級與 990 個人×文類級的語言特徵。已逐欄檢查三表，均缺可接回原文的回報停點及同一疑問的後續配對。Ness-Maddox 等人的補充資料與 Troemel 等人的 OSF 專案用無憑證方式得到 HTTP 401，自動審查拒絕使用受限的 view-only 存取。 [Christhilf 等人及可用範圍](https://doi.org/10.1080/0163853X.2025.2578594) · [RSAT 方法與資料](https://pmc.ncbi.nlm.nih.gov/articles/PMC9402089/) · [Ness-Maddox 等人](https://doi.org/10.1080/0163853X.2025.2577634) · [Troemel 等人](https://doi.org/10.1080/0163853X.2025.2591029)

**AmbiStory 提供後文改變詞義判斷的直接對照。** 測試有 155 個故事家族與 4,663 票。三種版本在歧義句末有完全相同的前綴，兩個結尾可以支持不同義項。可檢查已知歧義詞是否出現符合真人分布的語意更新。各版本由不同讀者組評分，無個人自覺疑問的起始及解除座標。 [資料、精確停點與待驗收規格](stage6/ambistory-context-update.md)

**Duff 等人的英文刺激提供跨句時序解讀的獨立觀測。** 48 套短文中，支持順時序的後續線索使真人選擇回溯時序的比例由 181/240 降到 30/240。另一組讀者在較晚線索版本有較低的時序判斷正確率，且呈現逐區段耗時差。作者起點表有 168 處早一詞，已用原題本與逐區段紀錄交叉核對。兩項實驗共用部分材料，每位讀者在同一篇至多做一次時序判斷。 [來源、重算與定位範圍](stage6/salt-discourse-audit.md)

## Currie 等人的跨句矛盾實驗

179 位六年級讀者閱讀每篇八句的英語短文，逐句按鍵前進，讀完後回答文章是否說得通。24 套矛盾材料各有近距離與遠距離版本，讓較晚的矛盾句和其前提相隔不同的句數。作者在發布檔中用底線標出前提及矛盾句。以兩組條件的 4,248 筆既有反應，估計讀者在篇末選「說得通」的機率。另 2,124 筆一致文章的資料可重算該條件的群體反應。出版檔案的對照文章缺少可直接把文章逐篇配回試次的相同識別鍵，因此模型擬合用 48 個明確編號的近／遠版本。48 個版本的矛盾句及其前一句都用原始逐詞毫秒換算獨立核對了字數，合計 96 個位置。 [原研究 Methods、表 2](https://eprints.lancs.ac.uk/id/eprint/147782/2/Main_Document_Unblinded_27Sept2020_toSubmit.pdf#page=9) · [公開刺激與資料](https://osf.io/sj28g/) · [資料稽核](results/stage5-semantic/currie-audit.json)

固定 Qwen3.5-2B 的最後層，用同一完整文章的因果前向取篇末及矛盾句末狀態。另做末句獨立輸入、矛盾句獨立輸入。模型接受正文，讀者背景為空。一次前向可取得多個已讀停點，輸出頭直接給機率。讀到矛盾句的讀出也用篇末評分作目標，衡量在該停點可用的訊息對最終判斷的關聯。作者標出的兩句是刺激位置，沒有個別讀者所選區段。各組按 24 套材料分六折留出，分開保留文體及兩個距離版本。同一讀者可以出現在不同材料折。

主要比較篇末的訓練折反應比例、獨立末句與完整文章三種估計。對 $i$ 筆刺激，以 $n_i$ 為真人反應數，$y_i$ 為選「說得通」的比例。凍結狀態做 L2 正規化，按 $n_i$ 除以折內訓練刺激的平均反應數加權，擬合係數 1 的 ridge，截距自由，預測裁切至 0–1。主要指標為逐試次 Brier，0 為理想，1 為最差。區間對完整材料家族作配對重抽樣。次要比較在字句完全相同、排列順序不同的材料，檢查矛盾句獨立讀出和該句可見前綴讀出如何估計同一篇末反應。 [前向與停點介面](stage5/currie_extract.py)

### 留出結果

**完整篇章和只看末句都比訓練折的平均回報比例準確。** 兩個文字條件之間的 Brier 差值很小，區間跨零。表中 Brier 按 4,248 筆既有判斷加權，越低越好。95% 配對區間按固定折及既有讀者樣本對 24 個材料家族重抽樣。 [六折預測與評分](results/stage5-semantic/currie-l4/scores.json)

| 讀出條件 | 留出 Brier | 相對訓練折平均的差值（95% 區間） |
| --- | ---: | --- |
| 訓練折平均回報比例 | 0.25283 | 參照 |
| 只看末句 | 0.23833 | −0.01451 [−0.02413, −0.00483] |
| 完整篇章 | 0.23797 | −0.01486 [−0.02531, −0.00452] |

完整篇章減只看末句的 Brier 差為 −0.00035，95% 區間 [−0.00349, +0.00265]。真人選「說得通」的比率在近、遠兩組為 47.32% 與 47.08%。材料等權的真人近減遠差為 +0.229 個百分點，完整篇章讀出為 +0.108 個百分點。23 個真人評分有高低的近／遠配對中，完整篇章對 11 個配對排出相同方向（47.8%）。 [配對結果](results/stage5-semantic/currie-l4/scores.json)

20 組只改句子排列的近／遠對照使用另外五折。只看矛盾句的 Brier 為 0.23820，讀到矛盾句時為 0.23884。差 +0.00065，95% 區間 [−0.00239, +0.00395]。相同矛盾句獨立輸入時每對都同分。在 19 個真人有高低的配對中，讀到該句的讀出有 10 個配對同方向（52.6%）。這些結果只涵蓋原研究的兩種矛盾距離，且每個評分都在篇末取得。 [精確重排配對](results/stage5-semantic/currie-l4/scores.json)

### L4 實測與重現

在 NVIDIA L4 以 FP32 擷取、FP64 擬合，共處理 98 個不同輸入和 6,656 個非 padding token。48 個矛盾句末與獨立截短前綴的 token 完全一致。六項截短與 padding 數值檢查通過，最大相對 RMS 差 $9.28\times10^{-7}$。擷取子程序耗時 50.55 秒，含載入、因果檢查及寫檔。擬合及評分子程序耗時 2.67 秒。從取得配置到送出停止請求約 94.56 秒，已確認釋放。帳戶四次取樣均顯示 767.96 CU，無法單獨由餘額辨識本次用量。 [擷取](results/stage5-semantic/currie-l4/extraction.json) · [執行與釋放](results/stage5-semantic/currie-l4/manifest.json) · [帳戶取樣](results/stage5-semantic/currie-l4/usage.jsonl)

資料可用 `research.stage5.currie_prepare` 從固定作者檔重建，兩個 JSONL 與稽核重跑逐位元組一致。在 L4 執行 `research.stage5.currie_extract` 及 `research.stage5.currie_fit --device cuda`。下載後用同一批特徵在本機重擬合，最大預測差 $1.11\times10^{-16}$，逐試次 Brier 也獨立重算一致。其他資料的描述統計可用 `research.stage5.inquisitive_audit`、`research.stage5.tedq_audit` 和 `research.stage5.hippocorpus_audit` 重跑。 [資料建置](stage5/currie_prepare.py) · [擬合與評分](stage5/currie_fit.py) · [獨立驗證](results/stage5-semantic/currie-l4/verification.json)

## Stage 5 驗收範圍

已重算公開資料中的句末圈選、兩句一停的疑問更新，以及跨句矛盾的篇末回報。跨句模型比較已在 24 套可精確配對的既有材料完成。語意疑問的圈選與後續更新兩個端點，各自仍需能對齊原文、可供模型使用的訓練與留出資料。句內首次觸發位置和疑問從何時起完全解除，在已核對的觀測中仍不可辨識。
