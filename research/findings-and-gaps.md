# CogLLM 階段回顧

截至 2026 年 11 月 21 日，以已封存的實驗、權重與介面驗證為準。研究執行暫停。

**目前有一個能運作的英文全文標色候選，以及幾個有真人標註留出證據的讀出。最核心的驗收仍開放。** 我們尚無法確認只貼入一篇文章時，模型輸出的疑問起點、所指上文與解除位置，可以共同重現同一讀者對同一疑問的歷程。已有的結果涵蓋不同觀察端點，應按各自的資料與比較解讀。 [原始完成條件](execution-plan.md#完成條件)

## 研究主張和目前交付

目標是讓固定的 Qwen3.5-2B 對正文及選填讀者背景做一次因果前向。讀出頭根據當下及已讀前綴的表徵，輸出可對回原文的反應分布、強度和所指區段。分析程式或使用者再解讀這些輸出。對同一位置換掉未讀後文，該位置的因果讀出應在數值容差內一致。空背景的結果依端點所定義的參考讀者群解讀。研究中的題幹、選項或預定命題只在對應的驗收中提供。 [數學規格](mathematical-formulation.md#1-輸入讀者與座標) · [研究方案](training-protocol.md)

| 整體驗收項目 | 狀態 | 目前的驗證範圍 |
| --- | --- | --- |
| 單次因果前向與全文標色 | 已通過原型驗收 | Gradio 貼文後同時顯示句色與詞底線，可查原文座標。切換或重新選取使用快取。已有實際 Colab L4 HTTP 流程和本機重播。 [驗證紀錄](gradio_app/README.md#核對實際模型與介面) |
| 空背景的閱讀分數 | 在限定端點通過 | 英文難詞順位、英文說明文篇章順位，以及一項摘要改寫效應有外部支持。獨立的逐句五級校準和自然篇章的逐 token 負荷強度仍待驗收。 [句級與篇章留出](stage-6-holdout-plan.md#同一-gradio-候選的英文留出結果) · [詞級留出](stage6/medreadme-lexical-results.md) |
| 同一疑問的逐位置軌跡 | 部分完成 | 已有首次回讀、逐句矛盾回報、指定片語的上文候選及指定問題的證據區間。三者在同一讀者和同一自發疑問上的連接仍缺主要驗收。 [資料與端點](semantic-tracking-first-run.md) |
| 選填讀者背景 | 實驗完成，增益未確立 | 背景留出已涵蓋語言、讀前測驗、教育及受控學習史。人群差幅多未通過驗收，App 正式入口目前只需正文。 [Kerwer](stage6/kerwer-reader-background-results.md) · [RACE](stage6/race-education-holdout-results.md) · [Beukers](semantic_tracking/beukers-background-results.md) |
| 同一最終 checkpoint 的跨研究驗收 | 尚未通過 | 目前已封存共用骨幹及各項讀出的 Gradio 候選。仍需以固定的同一套端點權重，在未用於設計的研究和材料上同時驗收反應分布、效應方向與幅度。 [候選清單](results/gradio-candidate-checkpoint/checkpoint.json) · [正式留出規則](stage-6-holdout-plan.md#獨立研究的正式驗收) |

這五項延續前次進度回報的計數。兩項已有明確範圍的驗收，其他三項有程度不同的實驗進展。從 Stage 0 到 Stage 6 都有實際產物，其中 Stage 6 的留出包含通過、未通過和資料不足的端點。 [各階段與封存紀錄](execution-plan.md#進度與驗收紀錄)

## 已驗證的結果和適用範圍

表中的 AP 為平均精確率，範圍 0–1，越高代表人工標記越集中在排序前段。MAE 是原量尺上的平均絕對誤差。NLL 是負對數概似，以 nats 計。MAE 和 NLL 越低越好，最低為 0。Spearman 相關衡量順位一致性，範圍 −1 至 1，越接近 1 表示正向順位一致程度越高。

| 問題 | 留出結果 | 可支持的主張 |
| --- | --- | --- |
| 哪些詞或片語較難？ | CompLex 1,094 筆目標的全量 MAE，單看目標 0.06988、讀到目標 0.08065、含完整句 0.08281。三種設計從 25% 增至全量都改善。同詞跨句的排序一致率只有約 51%–52%。 | 讀出可估指定詞或片語的群體平均難度。語境改變同一個詞難度時，這批結果的辨別力接近平手。 [CompLex](stage-5-complex-run.md#留出結果) |
| App 全文掃描能否找出難詞？ | MedReadMe 在 666 個可做句內排序的英語生醫句子，App 詞末讀出 AP 0.713，詞長 0.504、反頻率 0.565。23,286 個可評詞中有 23,284 個精確對齊。 | 支持受訓標註員圈選的生醫難詞與術語順位。同一詞末讀出在 CWI 專業新聞的句內詞對排序為 65.67%，詞長與詞頻合併為 75.25%，因此跨文類仍待驗收。 [MedReadMe](stage6/medreadme-lexical-results.md) · [CWI 專業新聞](core_features/cwi-news-holdout-results.md) |
| 句末表徵能否反映理解自評？ | PLABA-EVAL 因果句末基線理解自評 MAE 由 0.732 降到 0.616。五級讀出在開發折外也優於長度加版本基線，理解自評 NLL 差 −0.03489，95% 區間 [−0.05738, −0.01065]。 | 模型在句末的表徵含有對全文可見時真人句級自評有用的訊號。 [PLABA 基線](stage-3-first-run.md) · [五級分布](stage-6-holdout-plan.md#plaba-eval-句級評分分布) |
| 句分數能否用於篇章比較？ | CLEAR 的 410 篇留出中，將 App 句分數加入表面特徵，教師難度順位相關由 0.608 升至 0.691，差 +0.083 [0.051, 0.118]。說明文有增益，文學的區間跨零。Kerwer 的白話摘要對學術摘要差，真人 +0.547、換算後模型 +0.582 分，符合 ±0.5 分門檻。 | 支持部分英語篇章順位與摘要改寫的群體自評差。Kerwer 的小標差真人為 +0.359 分，模型約為 0，顯示即使同一份材料也需逐項核對版面與改寫效應。 [CLEAR](stage6/clear-reading-ease-results.md) · [Kerwer](stage-6-holdout-plan.md#同一-gradio-候選的英文留出結果) |
| 正文能否改善對真人作答的估計？ | 固定 OneStopQA 讀出換到 RACE 216 篇、1,940 筆開卷作答，提供正文後 NLL 由 1.3519 降至 1.0686。兩資料共用讀者；在實際答錯的 426 筆中，加入正文後的 NLL 由 1.3926 升到 1.4446。 | 支持指定開卷作答中正文的重要性。整體分數進步可能同時讓模型更難重現真人的錯誤選擇。 [RACE](stage-6-race-run.md#留出結果) |
| 找到哪裡開始回讀、往哪裡回讀？ | 在 16 組留出材料中，首次回讀的 AP 由位置與長度基線 0.066 升至 0.301。335 次有多個合法候選的回讀，其最遠目的地首選由 48.1% 升至 51.0%。 | 首次往回按鍵和最遠落點是行為端點。讀者可在察覺後才按鍵，也可帶著未解疑問繼續讀。 [首次回讀留出](semantic-tracking-first-run.md#首次主動回讀) |
| 已知片語或矛盾位置，能否回查上文？ | ARRAU 的 PEAR 20 篇、302 個可評查詢，前五命中由 23.3% 升至 34.4%，首選 9.5%。SCARECROW 已知矛盾位置的 226 筆留出中，上文區段 NLL 由 2.077 降至 2.027。 | 可將上文候選排序，有證據支持讓作者查看候選列表。實際閱讀者何時注意到指涉疑問尚需另外的回報。 [ARRAU](semantic_tracking/arrau-bridge-results.md) · [SCARECROW 上文](semantic-tracking-first-run.md#矛盾所指的較早文字) |
| 只有正文時，能否找出矛盾或消歧詞？ | Blott 2021 未見句框中，研究者指定消歧詞的首選命中從 39.1% 到 69.6%，實時提示的誤報門檻未通過。SCARECROW 的 246 篇留出中，109 篇可比的圈選 AP 為 Qwen 0.32559、表面 0.32566，嚴重度的增益區間亦跨零。 | Blott 支持有候選排序訊號。SCARECROW 是完整閱讀後的人工矛盾及程度圈選，在該設計中加入 Qwen 未改善篇內定位。 [Blott](semantic_tracking/blott-word-results.md) · [SCARECROW 程度](semantic_tracking/scarecrow-detection-results.md) |
| 指定問題的文字證據何時已可見？ | 新 QASPER 27 篇留出中，原疑問比同文交換疑問的區間 NLL 低 0.0830 nats，正式候選通過 2／5 項。ExpMRC 同文同停點逐 token 排序為 435／698，對稱校準後區間 NLL 3.076，位置基線 3.049。校準轉移到 Natural Questions 時，逐 token 是 27.210，未校準位置基線 4.507。 | 文字特徵能在部分資料區分同文不同問題。絕對到達機率對研究和缺標機制敏感。各資料量測的是已給題幹的人工文字證據。 [QASPER](qasper-factorial-run.md) · [ExpMRC](semantic_tracking/evidence-calibration-comparison-results.md) · [Natural Questions](semantic_tracking/nq-evidence-transfer-results.md) |
| 能否在整篇找到支持或反駁區段？ | ContractNLI 固定命題與原始候選邊界下，23 篇新文件的融合 AP 由 BM25 0.440 提升至 0.502，同篇命題配對由 0.701 提升至 0.809。換到獨立的生醫 Evidence Inference 留出，融合 AP 0.215，字面 0.306。 | 合約文件的證據順位有通過案例，跨領域和 App 自動分句後的候選幾何仍需各自驗證。 [合約排序](semantic_tracking/contractnli-rerank-results.md) · [App 座標](semantic_tracking/contractnli-app-geometry.md) · [生醫留出](semantic_tracking/evidence-inference-transfer-results.md) |

## 真人已量到的效應，模型仍未通過哪些

| 留出端點 | 真人對照 | 讀出與剩餘誤差 |
| --- | --- | --- |
| Frick 在相同目標句首次回報矛盾 | 20 組故事、1,800 條序列。不一致前文使首次回報增加 48.27 個百分點。 | 七特徵估 18.95，加詞表與注意力估 16.36。只加注意力的整篇 NLL 改善 −0.0299 nats，目標句幅度仍只估 20.01。 [結果](semantic_tracking/frick-signals-results.md) |
| Blott 消歧後的句末判斷 | 新書面句框上，真人「說得通」差 +16.33 個百分點。 | 用線索前後狀態可改善一項讀出對照的 NLL，配對監督估 +3.64 個百分點。 [結果](semantic_tracking/blott-update-results.md) |
| 捷克語歧義解除後，句末仍接受原先誤解 | 24 組未見材料、1,281 筆目標作答，真人差 +23.23 個百分點。 | 逐詞匯集加句末狀態估 +7.30。差值區間 [−23.60, −8.85] 超出 ±5 個百分點。固定英語消歧讀出也未重現第七詞的預定差。 [結果](semantic_tracking/czech-garden-results.md) |
| Rodger 同一讀者對同一政策的前後評分 | 301 人、1,806 組評分，真人負向個案相對控制差 −7.86 分，量尺 0–100。 | 正文讀出估 −1.21。加入同試次讀前評分估 −2.78，幅度仍未達標。 [結果](semantic_tracking/rodger-belief-results.md) |
| Oliver 同一讀者的詞彙提取與答案確認 | 14 個未見詞項、728 次作答。首次回報「話到嘴邊」相對「不知道」，限時內改按 KNOW 的差 +17.60，揭曉後確認的差 +38.61 個百分點。 | 估 +23.89 和 +42.74，幅度區間未通過 ±5 個百分點。整條按鍵的 NLL 1.81808，常數歷程基線 1.80160。 [結果](semantic_tracking/oliver-retrieval-results.md) |
| 讀者背景與同一文字的交互 | Kerwer 同時留出材料與讀者，有 1,947 筆。Beukers 用獨立重做樣本觀察不同學習順序。 | Kerwer 的讀前測驗與交互增益區間均跨零。Beukers 的獨立重做中，原 Qwen 交互讀出估兩組效應差 14.3 個百分點，真人為 78.0，幅度未通過。 [Kerwer](stage6/kerwer-reader-background-results.md) · [Beukers](semantic_tracking/beukers-background-results.md) |

〔推論〕不同端點的留出結果指向同一個驗收要求。排序、個別預測和群體效應幅度各有可能獨立失敗。Frick 的注意力訊號降低了一項整篇誤差，目標句的群體差幅仍明顯低估。要判斷一個讀出是否重現閱讀行為，需要分開查這三項，並保留長度、位置、詞頻及條件先驗等對照。 [Frick 分項結果](semantic_tracking/frick-signals-results.md#留出結果) · [真人效度規格](mathematical-formulation.md#真人效度)

## 訓練方法目前知道什麼

已登錄 46 篇實證文獻，其中 4 篇僅核到摘要或章節目錄。首批建置使用 PLABA-EVAL 句級自評及疑難、Zhan 華語續寫指涉、OneStopQA 開卷選答。後續擴充了既有詞彙、句級與篇章端點。可用訊號包含真人逐次標註、文獻條件差、已對齊的人工區間。每項保留其語言、可見材料、觀測粒度與分母。 [文獻觀測登錄](stage-1-observations.md) · [首批資料](stage-2-data.md)

已跑的主要方法是凍結 Qwen，監督擬合線性頭或共享 adapter，比較逐次損失、配對或區間排序、加入文獻效應損失。三任務共享加效應後，PLABA 理解 MAE 由 0.7556 降至 0.6963，Zhan 逐次選擇 Brier 由 0.2616 降至 0.2474。Brier 為機率與實際結果的平方誤差，越低越好。兩個文獻效應的幅度改善區間均跨零，Zhan 留出依然只有 +8.45 個百分點，真人 +24.46。後來把回讀、上文指向和弱監督解除聯合訓練，三個端點也未一起改善。 [聯合效應](joint-effects-run.md) · [語意三任務](semantic-joint-first-run.md)

第 12 層 logits、surprisal 和注意力已做過特徵比較。PLABA 句級疑難上，加入這些訊號未顯示穩定增益。Frick 只加注意力有一項整篇 NLL 改善，未解決效應幅度。Beukers 的回溯診斷則在固定故事模板中從表徵讀出作者指定的關鍵線索，接入按任務結構擬合的讀出後，兩組效應差由原 Qwen 交互讀出的 14.3 提升至 60.2 個百分點，真人為 78.0。這個結果提示讀出結構可能影響已存在的文字訊號能否轉成正確的行為幅度，仍需新材料檢驗。 [特徵對照](stage-3-feature-ablation.md) · [Beukers 線索診斷](semantic_tracking/beukers-background-results.md#線索辨識與選擇讀出的事後對照)

RL 和修改 Qwen 骨幹的 LoRA 都仍是設計中的對照選項。已取得的證據來自上述監督式讀出，尚無已驗收的 RL 或骨幹 LoRA 結果。即使一個損失能逼近文獻平均差，平均差本身也不提供讀者在每個 token 的疑問位置。 [訓練規格](training-protocol.md#訓練順序)

## 最關鍵的資料缺口

| 來源 | 目前真的看得到的事件 | 主要缺項 |
| --- | --- | --- |
| Frick | 逐句回報當前句是否與前文矛盾。首次回報可精確接到句末。 | 下一句改按一致在兩版都超過 92%，題目已換成下一句，因此沒有同一疑問的解除回報。 [反例與計數](semantic_tracking/frick-resolution-negative-control.md) |
| TED-Q | 同一提問者對同一疑問評量後文的回答程度，通常每兩句一次。 | 第一次達 3 分就停止追蹤，完全獲答是 5 分。依已呈現但未續評的停點，完全獲答比例可辨識範圍為 14.88%–36.32%。模型用途的逐字稿書面授權尚未取得。 [觀測界限](semantic_tracking/tedq-full-answer-bounds.md) · [使用條件查核](semantic_tracking/frick-resolution-negative-control.md#可直接核對的同疑問資料) |
| Onea／PITQ | 標註者對三篇故事留下 196 個行級情節問題，183 個有確認解答行。 | 標註是閱讀後回頭整理，確認解答後清除後續欄位，難以量到同一題重開。適合研究行級問題共存和首次閉合。 [標註規則與重計](semantic_tracking/onea-reopening-source.md) · [出現至解答的基線](semantic-tracking-first-run.md#故事情節疑問的解答位置) |
| 公開口述案例 | 可核對同一位讀者自發提問，之後在下一句或後續段落明示理解。 | 報告只刊登選錄，缺完整文字暴露時點、所有事件及未事件的分母。 [案例](semantic_tracking/reader-report-source-audit.md#同一讀者自發提問與確認理解的公開案例) |
| 前後或重複信念評分 | Rodger、Qiao、Albertazzi 和 Wahlheim 等有同一讀者對指定命題反覆評估的紀錄。Yazbec 有多主題文章與前後重答。 | 各研究的報告時間、文章位置與問題對照不同。Albertazzi 缺逐回合實際文字鍵。Wahlheim 在看完修正階段後才評，兩研究共用題材。Yazbec 部分前測重複且分數衝突，部分題號對照仍待釐清。 [Albertazzi 查核](results/albertazzi-narrative-source-audit.json) · [Wahlheim 查核](results/wahlheim-correction-source-audit.json) · [Yazbec 查核](results/yazbec-conceptual-change-source-audit.json) |

〔本研究發現〕嚴格「先解最新疑問」無法重現現有敘事觀測。Onea 的 115 個有一題結束、其他題仍開放的位置，79 個不符合嚴格後進先出規則。TED-Q 另有 13 次同一人先給較早問題 5 分、較新問題仍只有 1 或 2 分。這支持在設計上容許多題並存及不同關閉順序。重開、暫擱與讀者心中是否已完全理解，仍各需相應資料。 [Onea 結果](semantic-tracking-first-run.md#故事情節疑問的解答位置) · [TED-Q 核對](semantic_tracking/tedq-full-answer-bounds.md)

## App 目前能交付的畫面與剩餘驗收

原型已符合作者或編輯貼文後一眼看全文的需求。實際 L4 請求核對過四段、2,496 字元的文章，含 26 個句子標記與 425 個詞彙標記。一個詞不能精確對齊時保留原文。切換量尺不重跑模型。指定片語可回查最多五個上文候選，逐詞消歧檢視可顯示每句最多三個候選。這些色階和候選順位應按各端點的量尺及驗證範圍閱讀。 [全文預覽](results/gradio-combined-overview-l4/reading-overview.html) · [消歧檢視預覽](results/gradio-word-cue-route-l4/cue-preview.html) · [介面及驗收](gradio_app/README.md#實驗性逐詞與提問指向候選)

正式版尚需同一套封存讀出接通可驗證的疑問事件、對應上文、後續解答區間，以及有資料支持的選填讀者背景。規格另有逐前綴檢視、雙版本比較和研究匯出。現存的長文、指向、換文和消歧收據由數次實作驗收累積。正式發行仍需用最終版本一起重驗。 [原型規格](app-concept.md#結果畫面) · [候選權重契約](results/gradio-candidate-checkpoint/checkpoint.json)

## 恢復時可直接核對的三項研究驗收

1. **同一疑問的可觀測性。** 確認既有資料能把同一讀者、同一疑問、當時可見文字、回報位置、所指上文和後續狀態接在一起。缺值、未回報、觀測截止和已解除各有定義。測量只到句或兩句時，按原粒度驗收。這會決定能否檢查「句二開頭才卡住」一類逐 token 主張。 [資料查核](semantic_tracking/reader-report-source-audit.md) · [位置與座標規格](mathematical-formulation.md#3-平行讀出反應位置與強度)
2. **同一候選的外部效果。** 鎖定核心效應、量尺、容許差和未參與設計的來源家族，同時報逐次反應分布、幅度、定位與錯誤。包含一致性、背景同文對照，以及有證據與無法確認證據的樣本，避免只看一項平均增益。 [真人效度](mathematical-formulation.md#真人效度) · [獨立留出](stage-6-holdout-plan.md#獨立研究的正式驗收)
3. **研究輸出與正式 App 一致。** 同一 checkpoint、同一 tokenizer、同一頭和門檻，從貼文到原文座標、部分或完全解答區間及結果量尺走一次端到端驗收。把模型估計和事後對照的真人效應各自保留依據。 [完成條件](execution-plan.md#完成條件) · [App 驗證方式](gradio_app/README.md#核對實際模型與介面)
