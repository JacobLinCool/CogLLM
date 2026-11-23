"""Build the repository-scoped inventory of human reading evidence."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / '.cache/cogllm'
INDEX = ROOT / 'research/reading-evidence.md'
OUTPUT = Path(__file__).with_name('source-catalog.json')

# Readiness describes the recorded material and observation, independent of the
# model run split. Two publications or tasks can share a single material family.
STUDY_READINESS = {
    'R01': ('mcnamara_coherence', 'published_effect', '完整刺激和逐次反應待核，保留連貫與知識的交互對照。'),
    'R02': ('mcnamara_kintsch_history', 'published_effect', '已核對摘要，逐題材料和反應待取得。'),
    'R03': ('britton_repair', 'published_effect', '目前只核到研究摘要和部分說明，完整刺激及反應待取得。'),
    'R04': ('gilabert_causal', 'published_effect', '附錄及條件彙總可作重建線索，逐題刺激和作答對照尚未建置。'),
    'R05': ('van_silfhout_connectives', 'published_effect', '已登錄句後閱讀與橋接題端點，無本機逐次可對齊表。'),
    'R06': ('kendeou_newton', 'published_effect', '保留知識前測與反駁文本的條件差，逐試次對照未建。'),
    'R07': ('gu_medical_simplification', 'published_effect', '已登錄成人理解條件平均，完整原刺激與作答連結待核。'),
    'R08': ('gu_medical_simplification', 'shared_reanalysis', '重分析 R07，隨原研究共享材料和作答家族。'),
    'R09': ('tibken_inconsistency', 'missing_stimuli', '公開分析表有偵測及選句合併正確性，所選句索引和完整刺激缺。'),
    'R10': ('beker_text_explanations', 'published_effect', '兩個實驗的前篇解釋對照已登錄，無本機可訓練逐次表。'),
    'R11': ('chi_self_explanation', 'published_effect', '保留自我解釋干預與知識前後測，缺模型使用的同次資料。'),
    'R12': ('mcnamara_istrat', 'published_effect', '兩種訓練條件及分題型結果，原刺激與逐題資料待核。'),
    'R13': ('recht_baseball', 'published_effect', '知識與能力分組的回憶，缺可對齊的原始讀者反應。'),
    'R14': ('ozuru_biology_cohesion', 'published_effect', '材料與題型分開登錄，無本機對照後的逐次反應。'),
    'R15': ('mckeown_history', 'published_effect', '保留寫法和回憶與理解的群體差，逐筆資料待核。'),
    'R16': ('tibken_inconsistency', 'missing_stimuli', '同類說明文矛盾範式，跨研究原文重疊尚待核對。'),
    'R17': ('wiley_rayner_topic', 'published_effect', '保留標題影響下游名詞的效應，無可接的逐次觀測。'),
    'R18': ('hyona_lorch_headings', 'published_effect', '保留雙篇同正文小標效應，完整條件刺激與個別反應待核。'),
    'R19': ('harp_mayer_details', 'published_effect', '圖文教學材料與回憶遷移端點，純文字反應另需觀測。'),
    'R20': ('bai_reading_policy', 'public_unprepared', '有作者資料入口，限時、可見文字、行動與理解欄位尚待接入。'),
    'R21': ('albrecht_obrien_consistency', 'published_effect', '保留一致性操弄作設計對照，試次材料尚未建置。'),
    'R22': ('wang_global_coherence', 'published_effect', '已核對摘要與操弄方向，未取得可對齊逐次資料。'),
    'R23': ('troyer_domain_knowledge', 'published_effect', '已登錄知識和 N400 端點，試次鍵、預處理與刺激尚未建置。'),
    'R24': ('schotter_masked_reread', 'published_effect', '屏蔽回讀與篇末理解的介入已登錄，無本機逐次對照。'),
    'R26': ('demberg_sayeed_pupil', 'public_unprepared', '有出版資料庫入口，逐字及生理時間窗尚待對齊。'),
    'R27': ('sap_syntactic_ambiguity', 'public_unprepared', '眼動與自定步速沿用相同 SAP 材料，整組刺激共同分割。'),
    'R28': ('hirotani_punctuation', 'published_effect', '標點和處理位置的實驗對照已登錄，原始眼動未建置。'),
    'R29': ('xue_sentence_boundaries', 'public_unprepared', '有作者資料入口，字詞邊界及逐次眼動尚待核。'),
    'R30': ('smith_levy_surprisal', 'published_effect', '詞機率及閱讀時長的文獻對照，底層語料需另按材料家族準備。'),
    'R31': ('christianson_role_errors', 'published_effect', '已核對句法角色答錯類別，完整可重現試次對照待核。'),
    'R32': ('camblin_local_global', 'published_effect', '眼動／ERP 的局部與廣域對照，未建完整逐次輸入。'),
    'R33': ('claus_kelter_flashbacks', 'published_effect', '保留文字順序與故事時間的操弄，完整逐次資料未建。'),
    'R34': ('cozijn_causal_relations', 'published_effect', '已登錄連詞、語段時間與推論驗證端點，無本機逐次對照。'),
    'R35': ('kleijn_connectives', 'published_effect', '完整荷語文本及各關係的逐次克漏作答待核。'),
    'R36': ('zhan', 'aligned_reading', '作者固定版已重建逐次指涉反應及來源家族。'),
    'R37': ('wu_mo_antecedent_distance', 'published_effect', '已登錄中文版前提距離效應，逐次材料與可見位置待核。'),
    'R38': ('scholman_chinese_connectives', 'public_unprepared', '有作者資料目錄，可作閱讀時間及未重現促進的對照，尚未接入。'),
    'R39': ('yu_hsiao_lin_pragmatics', 'published_effect', '僅核到摘要或原刊目錄，完整情境和逐次選擇待取得。'),
    'R40': ('wei_ba_construction', 'public_unprepared', '有資料入口，眼動與自定步速的共同材料須先對齊。'),
    'R41': ('plaba', 'aligned_reading', '已重建逐句自評、原文改寫配對與試次，同 D09。'),
    'R42': ('daneman_carpenter_span', 'published_effect', '保留早期讀者能力檢驗，沒有可對應文章的讀者逐次資料。'),
    'R43': ('yeung_explanatory_notes', 'published_effect', '保留註解、讀者族群與理解效應，原始刺激及反應待核。'),
    'R44': ('love_mckoon_gerrig', 'published_effect', '角色與主文的關係操弄已登錄，完整探針材料和逐次反應待核。'),
    'R45': ('jones_bergen', 'public_unprepared', '已核對作者刺激欄及兩實驗逐次資料，排除規則、版本與用途待收據。'),
    'R46': ('kendeou_refutation_reactivation', 'published_effect', '同時保留再活化與理解干擾，全文條件與逐次反應待取得。'),
    'R47': ('pettijohn_event_transition', 'published_effect', '附錄示例可核，完整刺激及逐次意外自評未接入。'),
    'D01': ('onestop', 'aligned_reading', '對齊作答和眼動回讀，共用新聞及簡化版本。'),
    'D02': ('tibken_inconsistency', 'missing_stimuli', '同 R09 的發表分析表，缺完整呈現刺激及所選句欄位。'),
    'D03': ('reco', 'public_unprepared', '專家推論與證據標記屬輔助資料，正式建置與 D12 書評資料分開。'),
    'D04': ('qasa', 'public_unprepared', '專家問答與證據供輔助編碼，完整試次及原文尚未準備。'),
    'D05': ('wikidomains', 'public_unprepared', '概念候選及改寫評估可作弱標，逐篇使用條件待核。'),
    'D06': ('expmrc', 'aligned_auxiliary', '標註的是題目的支持範圍，沒有首次察覺答案位置。'),
    'D07': ('potec', 'rights_pending', '數值與背景已對齊，出版社教材的雲端使用範圍未建立。'),
    'D08': ('meco_traditional_chinese', 'public_unprepared', '繁中篇章理解與行向操弄待原檔、刺激及試次對齊。'),
    'D09': ('plaba', 'aligned_reading', '同 R41，既有句級、疑難類型與篇末作答重建。'),
    'D10': ('inquisitive', 'missing_stimuli', '公開問題及圈選位置，完整合法正文和無事件分母待補。'),
    'D11': ('tedq', 'rights_pending', '疑問與再評時間可重建，TED 逐字稿的 AI 用途需專門授權。'),
    'D12': ('reading_between_lines', 'rights_pending', '全文圈選存在，書評原文與訓練使用範圍待查。'),
    'D13': ('im_farber_simplification', 'public_unprepared', '原始備份與正式逐例分析映射尚未確認，原文用途待核。'),
    'D14': ('complex', 'aligned_reading', '官方英語詞／片語負荷彙總已對齊指定位置。'),
}

# dataset id, status, checked scope, cache locations, source receipt, endpoint
DATASETS = (
    ('plaba', 'aligned_reading', '句級評分與白話改寫、結構疑難、讀後自評及開卷反應。作答證據依已訂正的版本。',
     'stage2/plaba|expository-lora/prepared/plaba', 'stage-2-data.md', 'sentence_ratings|sentence_reports|comprehension'),
    ('onestop', 'aligned_reading', '正文理解反應與閱讀回視分別保存。兩端點按共用新聞、基礎版或簡化版分組。',
     'stage2/onestop|onestop-rereading/assay-v1', 'stage6/onestop-rereading-audit.md', 'comprehension|rereading'),
    ('zhan', 'aligned_reading', '作者發布的華語續寫逐次資料已按材料對齊。不同提示或語法臂保持同家族。',
     'stage2/zhan', 'stage-2-data.md', 'reference_choice'),
    ('complex', 'aligned_reading', '作者發布指定詞或片語的多位真人閱讀難度彙總。',
     'backbone-transfer/prepared/complex|stage5/complex', 'stage-5-complex-run.md', 'lexical_difficulty'),
    ('cwi', 'aligned_reading', '指定詞的語境複雜度判斷，作者規則決定負例與可用分母。',
     'core-features/cwi|core-features/cwi-lexical-readout/prepared-v2|core-features/cwi-news-holdout/prepared',
     'core_features/cwi-lexical-readout-results.md', 'lexical_difficulty'),
    ('frick', 'rights_pending', '前綴一致性與首次矛盾按鍵已對齊。論文授權與獨立工作表及刺激的使用條件分開核。',
     'semantic-tracking/frick-consistency/prepared|backbone-transfer/prepared/frick-both',
     'semantic_tracking/frick-consistency-source.md', 'first_inconsistency_report'),
    ('currie', 'aligned_reading', '依作者材料提供敘事與說明文的一致性判斷與閱讀時間，判斷的時機依觀測表。',
     'stage5/currie|stage5/currie-source', 'stage-5-semantic-run.md', 'coherence_judgment|reading_time'),
    ('larrc', 'aligned_reading', '獨立研究使用文章層級的一致性判斷；巢狀情境和本篇結束分開。',
     'semantic-tracking/larrc-consistency/prepared', 'semantic_tracking/larrc-consistency-source.md', 'coherence_judgment'),
    ('kerwer', 'aligned_reading', '作者資料限定科學用途。保留原摘要家族、小標臂、篇末反應及已核讀前變項。',
     'kerwer-2021/holdout|kerwer-2021/reader-background-v1|expository-lora/prepared/kerwer',
     'stage6/kerwer-article-holdout.md', 'article_comprehensibility|background'),
    ('clear', 'aligned_reading', '既有對照僅使用逐列來源授權符合條件的材料，保留同一來源的家族分割。',
     'clear-reading-ease/prepared', 'stage6/clear-reading-ease-source.md', 'article_reading_ease'),
    ('ambistory', 'aligned_reading', '多版本故事與序位真人評分，保留線索、篇章狀態與原作者家族分割。',
     'ambistory/causal-v3|ambistory/effect-inputs/prepared', 'stage6/ambistory-context-update.md', 'interpretation_rating'),
    ('first-readback', 'aligned_reading', '英德原顯示區段的首次回讀，依原資料建風險集合。與其他 Paape 資料檢查重合。',
     'semantic-tracking/first-readback-v1|core-features/readback|semantic-tracking/reanalysis',
     'semantic_tracking/first-readback-transfer.md', 'first_rereading'),
    ('context-readback', 'aligned_reading', '正式英文同片段前文操弄，已做失配診斷。作者資料為 CC BY-NC 4.0。',
     'semantic-tracking/context-readback-v1|semantic-tracking/context-6uv37-source',
     'semantic_tracking/context-readback-transfer-results.md', 'first_rereading|context_effect'),
    ('czech', 'rights_pending', '刺激與誤解作答已建置。文章具 CC BY 且作者指向公開原始材料，獨立 OSF 資料發行的授權欄空白。',
     'semantic-tracking/czech-garden-path/prepared', 'semantic_tracking/czech-garden-path-source.md', 'lingering_interpretation'),
    ('blott', 'aligned_reading', '2021 及 2023 OSF 專案皆為 CC BY 4.0。兩年句框有共用詞彙關聯，依已核重用範圍分組。',
     'semantic-tracking/blott-ambiguity/prepared|semantic-tracking/blott-update/2021/prepared|reading-trace-source-audit/blott-rerun',
     'semantic_tracking/blott-update-results.md', 'coherence_judgment|disambiguation'),
    ('oliver', 'aligned_reading', 'CC0-1.0 Borealis v3 的定義後回報、提示、揭曉後判斷。定義保留原有來源致謝。',
     'semantic-tracking/oliver-retrieval/prepared', 'semantic_tracking/oliver-retrieval-results.md', 'retrieval_path'),
    ('act', 'rights_pending', '公開檔有原始線索與首次答對或未答對，獨立 OSF 檔未列授權。',
     'semantic-tracking/act-german/assessment-v1|semantic-tracking/act-german/material-v1',
     'semantic_tracking/act-source.md', 'first_correct_answer'),
    ('core-vocabulary', 'aligned_reading', '提供詞項逐次支持與可測的邊際，未知的個體間路徑保持不可辨識。',
     'semantic-tracking/core-vocabulary/marginal-disjoint-panel-v1', 'semantic_tracking/core-vocabulary-source.md', 'retrieval_marginals'),
    ('scarecrow', 'aligned_auxiliary', '作者釋出供研究使用。端點限全文事後真人圈選，既有整備只送模型續文，第三方新聞前綴留在本機。',
     'semantic-tracking/scarecrow-detection/prepared', 'semantic_tracking/scarecrow-detection-results.md', 'annotation_span'),
    ('onset-source', 'aligned_auxiliary', '來源是人類圈選與先前建置的對齊評分。依原標註方式呈現，勿改寫為首次閱讀反應。',
     'semantic-tracking/source|semantic-tracking/remote-input', 'semantic-tracking-first-run.md', 'annotation_span'),
    ('onea', 'aligned_auxiliary', '故事中的問題與關閉行，依公開故事資源可用範圍取座標。原生端點為文本標記的解答。',
     'semantic-joint/closure|semantic-tracking/resolution-osf', 'semantic_tracking/onea-reopening-source.md', 'annotated_closure'),
    ('beukers', 'aligned_reading', '以作者試次及共同段落模板重建學習背景、證據呈現及讀者選擇。變體對照按共用模板分組。',
     'beukers/assay-v1|beukers/inserted-assay-v1', 'semantic_tracking/beukers-background-results.md', 'background_interaction|choice'),
    ('rodger', 'aligned_reading', '同一讀者的政策事前、事後評分與文字干預版本可對齊。',
     'semantic-tracking/rodger-belief/prepared', 'semantic_tracking/rodger-belief-source.md', 'belief_update'),
    ('qiao', 'aligned_reading', '同一讀者循序判斷的條件及可見證據已建置，組別資訊另存。',
     'semantic-tracking/qiao-evidence/prepared-v2', 'semantic_tracking/qiao-evidence-source.md', 'belief_update'),
    ('wahlheim', 'public_unprepared', '公開資料與素材已在本機，修正與揭示之後的試次需逐段對齊。兩輪共享主題。',
     'semantic-tracking/wahlheim-correction-source', 'results/wahlheim-correction-source-audit.json', 'belief_update|memory'),
    ('albertazzi', 'missing_stimuli', '真人重複置信評分公開，當次完整刺激與題目版本對照不全。',
     '', 'results/albertazzi-narrative-source-audit.json', 'belief_update'),
    ('yazbec', 'source_unaligned', '原發布部分重複前測有相異分值，題號與讀者原始作答未全部唯一對齊。',
     '', 'results/yazbec-conceptual-change-source-audit.json', 'conceptual_change'),
    ('anglin', 'source_unaligned', '本機有 SAV 與文稿，Study 1 的呈現文字、試次和問卷需要依稽核保留限定。',
     'anglin', 'anglin-study1-audit.md', 'belief_revision'),
    ('textcomplexityde', 'aligned_reading', '德文逐句閱讀評分。取經過收據檢查的固定版本與版權註記。',
     'textcomplexityde/transfer-v2', 'stage6/textcomplexityde-transfer-amendment.md', 'sentence_ratings'),
    ('studentannotations', 'aligned_reading', '德國高中生五級句級理解難度與讀前語言背景，職涯指南子集已排除和 TextComplexityDE 重疊的 554 句。作者提供研究用途。',
     'studentannotations/guide-v1', 'stage6/student-annotations-background.md', 'sentence_ratings|background'),
    ('turkish', 'aligned_reading', '土耳其語相同長度的句對比較與讀後題目，保留底層摘錄的授權範圍。',
     'turkish-v5/assay-v1', 'stage6/turkish-sentence-source.md', 'pair_preference|comprehension'),
    ('nahatame', 'rights_pending', '已對齊原測驗的英語閱讀、篇末難度及讀前能力，EIKEN 全文的模型抽取使用範圍未核。',
     'nahatame-t75nh/prepared-v2', 'nahatame-teco-data-audit.md', 'comprehension|proficiency'),
    ('medreadme', 'aligned_auxiliary', '專家圈定難字課題及對照槽；預測詞級專家標記屬輔助目標。',
     'medreadme-audit/lexical-holdout-v3', 'stage6/medreadme-span-audit.md', 'annotation_span'),
    ('kangoo', 'aligned_reading', '候選詞的真人難度報告依本地查核的原始資格與句界。',
     'core-features/kangoo-v4/local-audit', 'core_features/kangoo-source-audit.md', 'lexical_difficulty'),
    ('race', 'aligned_reading', 'Prolific 發行有 1,940 筆真人逐次選答。沿用 RACE 非商業研究條件，與 OneStopQA 共用 970 位讀者。',
     'stage6/race', 'stage-6-race-run.md', 'comprehension'),
    ('qasper', 'aligned_auxiliary', '論文問題和事後支持段落。人標證據抵達可作輔助區間，原提問者的察覺時刻未測。',
     'qasper/input|qasper/dev-holdout|qasper/censor-v1', 'qasper-evidence-run.md', 'annotation_evidence'),
    ('iirc', 'aligned_auxiliary', '問題及跨文章支持標註可評文字證據關聯與時點，與人類當下反應分開。',
     'iirc/linked-identity-v2|iirc/linked-evidence-v1', 'iirc-linked-evidence-protocol.md', 'annotation_evidence'),
    ('expmrc', 'aligned_auxiliary', '中英文標註的讀解證據範圍，按共享文章與題目 split。',
     'exp-mrc/v1|semantic-tracking/same-question-evidence/expmrc-v1', 'semantic_tracking/expmrc-transfer.md', 'annotation_evidence'),
    ('evidence-inference', 'aligned_auxiliary', '人標科學文獻證據，訓練須逐篇納入已核的文本授權。與有人實際回應的閱讀端點分開。',
     'semantic-tracking/evidence-inference-transfer/prepared', 'semantic_tracking/evidence-inference-source.md', 'annotation_evidence'),
    ('contractnli', 'aligned_auxiliary', '人標全文合約判斷及證據，屬判斷和定位標註，無讀者當下狀態。',
     'semantic-tracking/contractnli-prepared', 'semantic_tracking/contractnli-evidence-source.md', 'annotation_evidence'),
    ('natural-questions', 'aligned_auxiliary', '人標文章回答範圍。題目在閱讀輸入，證據定位為輔助端點。',
     'natural-questions/holdout-v1', 'semantic_tracking/nq-evidence-source-audit.md', 'annotation_evidence'),
    ('gum-gentle', 'aligned_auxiliary', '人標橋接先行詞，GUM 與 GENTLE 不是同一語料。版號和文字授權依收據。',
     'gum-bridge/prepared|gum-bridge/gentle', 'semantic_tracking/gum-bridge-results.md', 'annotation_reference'),
    ('arrau', 'aligned_auxiliary', '人標橋接與先行詞，原標註單位與讀者當下判斷不同。',
     'arrau-bridge/prepared', 'semantic_tracking/arrau-bridge-results.md', 'annotation_reference'),
    ('ncloze', 'aligned_reading', '既有讀者填字反應可比較接續線索，使用資料所保留的顯示單位。',
     'semantic-tracking/ncloze/raw', 'semantic_tracking/ncloze-source.md', 'cloze_choice'),
    ('qanta25', 'source_unaligned', '已核對語音與搶答欄位。如實區分跟讀式提示、回答與沒有完整位置的觀測。',
     'semantic-tracking/qanta25-source-audit', 'semantic_tracking/qanta25-source.md', 'quiz_buzz'),
    ('hippocorpus', 'public_unprepared', '既有故事重述和事件標記，原發行與各版標記需按同故事分組。',
     'raw/hippocorpus', 'semantic_tracking/semantic-load-curve-plan.md', 'event_annotations'),
    ('lam-hwang', 'public_unprepared', '原始留出 dir 已有部分材料。先核對版本是否與已用研究共享正文。',
     'raw/lam-hwang-holdout', 'stage-6-holdout-plan.md', 'reference_choice'),
    ('abel', 'missing_stimuli', '順序對照已登錄，可取得的刺激是英譯且缺逐段反應。原研究估計可作研究層級對照。',
     'abel-interleaving/raw', 'stage6/abel-interleaving-audit.md', 'order_effect'),
    ('french-connectives', 'source_unaligned', '法語連接詞資料已本機稽核，題型、同批刺激及分割鍵依實驗另確認。',
     'french-connectives-audit/raw', 'stage6/french-connectives-audit.md', 'connective_response'),
    ('salt-discourse', 'rights_pending', '原始刺激、實驗一事件先後選答、實驗二逐區段時間及篇末作答可對齊。兩實驗共用材料，OSF 與 DataCite 未列授權。',
     'salt-discourse-audit', 'stage6/salt-discourse-audit.md', 'discourse_processing'),
    ('yadav', 'aligned_reading', 'CC BY 4.0 的德荷逐試次難度與受控嵌套版本已依原作者鍵核對到顯示刺激。全句及逐詞兩實驗共用大部分原題。',
     'yadav/raw', 'stage6/yadav-transfer-audit.md', 'discourse_reading'),
    ('potec', 'rights_pending', '資料表與數值可供本機數值統計；已發現教材再授權限制，完整正文的雲端使用缺依據。',
     'potec-terms-audit', 'stage6/potec-background-readiness.md', 'comprehension|background|eye_tracking'),
    ('sap_syntactic_ambiguity', 'public_unprepared', '先封同材料自定步速與眼動家族，再核對可重建的閱讀位置及理解題。',
     '', 'stage-1/measurement-and-boundaries.md', 'reading_time|eye_tracking|comprehension'),
    ('scholman', 'public_unprepared', '原論文指向公開的中文連詞自定步速資料。實驗刺激及第二標記原位置須依作者檔確認。',
     '', 'reading-evidence.md', 'reading_time|connective_effect'),
    ('christianson_masking_2024', 'public_unprepared', '眼動遮罩與篇末理解是獨立對照候選。作者檔案可見性及逐試次對齊待核。',
     '', 'expository-reading-plan.md', 'rereading_policy|comprehension'),
    ('cedar_enro', 'source_unaligned', '已檢查作者眼動及作答表。能否取得正文與同一試次的連結需另行清查。',
     'enro-audit', 'stage-6-holdout-plan.md', 'eye_tracking|comprehension'),
)


def sha256(path):
    with path.open('rb') as source:
        return file_digest(source, 'sha256').hexdigest()


def literature_inventory():
    entries = []
    for line_no, line in enumerate(INDEX.read_text(encoding='utf-8').splitlines(), start=1):
        match = re.match(r'^\| \*\*([RD]\d+)\s*([^*]*)\*\*\s*([^|]*)\|\s*([^|]*)\|', line)
        if not match:
            continue
        entry_id, inside, outside, observation = match.groups()
        family, status, reason = STUDY_READINESS[entry_id]
        entries.append({
            'id': entry_id,
            'title': (inside + ' ' + outside).strip(),
            'source': f'research/reading-evidence.md:{line_no}',
            'family': family,
            'status': status,
            'reason': reason,
            'observation': observation.strip(),
            'links': re.findall(r'\]\((https?://[^)]+)\)', line),
        })
    if len(entries) != len(STUDY_READINESS) or {entry['id'] for entry in entries} != STUDY_READINESS.keys():
        raise ValueError('The evidence index and readiness decisions differ.')
    return sorted(entries, key=lambda entry: entry['id'])


def dataset_inventory():
    entries = []
    for name, status, reason, cache, reference, signal in DATASETS:
        reference = 'research/' + reference
        if not (ROOT / reference).is_file():
            raise FileNotFoundError(reference)
        relative = cache.split('|') if cache else []
        available = [path for path in relative if (CACHE / path).exists()]
        if status.startswith('aligned') and len(available) != len(relative):
            raise ValueError(f'An aligned source is missing from the catalog snapshot: {name}')
        entries.append({
            'id': name, 'status': status, 'reason': reason, 'reference': reference,
            'cache': relative, 'available': available, 'signals': signal.split('|'),
        })
    if len({entry['id'] for entry in entries}) != len(entries):
        raise ValueError('Dataset identifier repeated.')
    return sorted(entries, key=lambda entry: entry['id'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = {
        'schema_version': 1,
        'scope': 'CogLLM 已登錄的 46 篇實證論文、14 個資源及後續在本機或研究紀錄中可追查的資料。'
                 '準備狀態與模型工作批次的資料切分分開。檔案存在只代表已取得對應路徑。',
        'source_index_sha256': sha256(INDEX),
        'literature': literature_inventory(),
        'datasets': dataset_inventory(),
        'shared_split_constraints': [
            {'families': ['onestop', 'race'], 'axis': 'readers',
             'evidence': 'research/stage-6-race-run.md', 'scope': '同一次 Prolific 實驗共用 970 位讀者。'},
            {'families': ['race', 'expmrc'], 'axis': 'materials',
             'evidence': 'research/semantic_tracking/expmrc-transfer.md',
             'scope': 'RACE+ 的兩篇來源有已核重疊。新增版本仍需核對文章。'},
            {'families': ['studentannotations', 'textcomplexityde'], 'axis': 'materials',
             'evidence': 'research/stage6/student-annotations-background.md',
             'scope': '職涯指南準備版本已剔除既有 TextComplexityDE 的 554 句。'},
            {'families': ['first-readback', 'context-readback'], 'axis': 'materials',
             'evidence': 'research/semantic_tracking/context-readback-transfer-results.md',
             'scope': '先前版本聲明兩個訓練來源無已查全文或片段重合。新的材料納入時需重查。'},
        ],
    }
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')
    print(f"{len(output['literature'])} literature and resource entries, {len(output['datasets'])} dataset entries")


if __name__ == '__main__':
    main()
