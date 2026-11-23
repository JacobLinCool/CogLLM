# CogLLM 全文閱讀分析

貼入正文即可在同一篇原文看見句子底色和詞彙底線。句子底色預設顯示模型估計的閱讀容易度，可切換成模型估計的理解自評，兩者都是 1–5 分。詞彙底線呈現詞末 token 的模型估計，範圍是 0–1，1 分最難。底色與底線都用篇內相對色階。暖色代表篇內預估較難，或預估理解自評較低，冷色代表相反方向。滑鼠停在詞上可看分數及座標。展開逐句明細可查句子的 Unicode 字元範圍，從 0 起算，區間不含終點。

預設每句模型估計只使用讀到該句句末的文字，參照的是 PLABA-EVAL 英語生醫摘要及其白話改寫在全文可見時的真人群體自評。跨領域和其他語言屬實驗性。預設全文檢視在每個空白行分隔的段落內使用英文 Punkt 分句，句界可能與資料集中的人工句界不同。 [基線與量尺](../training-protocol.md#已完成的首輪基線) · [正式 app 完成條件](../execution-plan.md#完成條件)

## 在 Apple Silicon 啟動

使用可存取 Metal 裝置的本機終端機、Python 3.11 以上及本機的官方 Qwen3.5-2B 快照。以下命令從 repo 根目錄執行。預設使用隨 repo 封存的候選 checkpoint，包含英文分句參數及每項讀出的來源收據。

```sh
uv venv --python 3.11 .venv-cogllm
uv pip install --python .venv-cogllm/bin/python -r research/gradio_app/requirements.txt
.venv-cogllm/bin/python -m research.gradio_app.checkpoint verify
.venv-cogllm/bin/python -m research.gradio_app.app --model /path/to/15852e8c16360a2fea060d615a32b45270f8a8fc
```

在瀏覽器開啟 `http://127.0.0.1:7860`。App 載入候選 checkpoint，並核對上文讀出所綁定的 checkpoint ID、tokenizer 與投影。所有讀出共用同一份 Qwen 快照及一次正文前向。可用 `--checkpoint` 指定同一候選的封存目錄。「分數來源與適用範圍」顯示使用的 checkpoint ID 和上文讀出版本。 [預設 checkpoint 封存清單](../results/gradio-candidate-checkpoint/checkpoint.json) · [上文讀出封存](../results/arrau-bridge-l4/reference-release.json)

`--causal-plaba` 與 `--full-context-plaba` 可重現先前的單讀出實驗。這兩個診斷模式使用 NLTK 已安裝的英文 `punkt_tab`。三篇驗證文章在因果句末的本機分數與 L4 最大差為 0.00321 分。 [因果權重與指標](../results/causal-plaba-checkpoint/manifest.json) · [本機驗證](../results/causal-plaba-checkpoint/local-verification.json)

## 實驗性逐詞與提問指向候選

各項分數共用一次正文前向。「閱讀容易度」與「理解自評」同時顯示詞彙底線。「原文標記」中的「詞彙難度」可單獨比較詞彙，以底色標出可對齊的詞。這些切換會使用同一份已分析結果。「提問指向候選」假設已有提問，依各段末可見文字排序本段可能被問及的片語。選取片語可查看候選範圍及用於計分的可見停點。 [CompLex 留出重播](../results/stage5-complex/lexical-runtime-replay.json) · [IIRC 片語留出](../iirc-anchor-run.md)

在「原文標記」切換到「消歧線索候選」，即可在正文看見每句最多三個候選。顏色標出句內名次。滑鼠停在候選詞上可查看排序分數、原文字元範圍與詞末 token 索引。原始分數只使用截至該詞的文字，名次按完整句內的可評詞排序。分數採相對排序量尺。只有當詞與前一詞的詞末都能精確對齊時才計分。此檢視重用正文前向，切換時由快取取值。 [讀出封存](../results/blott-word-l4/word-cue-release.json)

定位的真人研究範圍是 Blott 英語實驗句。2021 年以句框留出時，研究者標記的線索排第一的比例是 69.6%，字面對照是 39.1%。第一時間提示的誤報門檻未通過，2023 年的真人句末理解差幅也未重現。貼入一般篇章時，候選可作為人工檢查的提示，讀者實際在哪裡察覺或解開疑問仍需逐位置觀測。 [逐詞定位、首次提示與真人按鍵結果](../semantic_tracking/blott-word-results.md)

本機以封存表徵重播 88 句、1,073 個候選，字元座標、token、名次與原始分數完全一致。實際 L4 Gradio HTTP 驗收涵蓋兩篇正文，每篇各一次 Qwen 前向。六次快取切換共用已分析結果，增加零次前向。後一篇加上含 CRLF 與 Unicode 的段落，保留前一篇句子的 10 個候選和座標，最大原始分數差為 $6.8\times10^{-7}$。另對本機 HTTP 預覽驗證桌面 1,280 px、手機 390 px 的排列與鍵盤聚焦。 [實際 L4 標色預覽](../results/gradio-word-cue-route-l4/cue-preview.html) · [L4 HTTP 驗收](../results/gradio-word-cue-route-l4/verification.json) · [本機重播與 HTTP](../results/gradio-word-cue-route-l4/local-verification.json) · [瀏覽器收據](../results/gradio-word-cue-route-l4/browser/ui-visual.json)

雙層檢視以同一候選 checkpoint 在 Colab L4 通過實際 Gradio HTTP 驗證，兩種句級檢視都在同一份原文保留 26 個句子標記、425 個詞彙標記及其 token 和字元座標。一個詞的座標未能精確對齊，原文字串保留。正文分析做一次因果前向，三次切換額外做零次前向。桌面 1,280 px 與窄螢幕 390 px 另以合成狀態核對底線樣式及無橫向溢出。 [L4 驗證](../results/gradio-combined-overview-l4/verification.json) · [實際回應 HTML](../results/gradio-combined-overview-l4/reading-overview.html) · [釋放紀錄](../results/gradio-combined-overview-l4/run-verification.json)

CompLex 的真人留出標註針對指定詞或片語。以留存的 L4 特徵重播 1,094 筆指定目標，平均絕對誤差為 0.08065 分。全篇詞彙掃描與使用跨段前文產生的提問指向候選均屬實驗性。 [外推範圍](../stage-5-complex-run.md#留出結果)

要回查一個指涉，先分析正文，再展開「查指定片語的上文」。貼入原文片語並按「查前文」。若片語出現多次，可依原文片段和字元位置選取。畫面以黃底標出指定片語，以藍底及名次在全文標出最多五個前文候選詞。查詢使用同一分析已保存的 128 維投影。從「上文指向」重新分析正文時，會清除舊查詢並回到句底色和詞底線的閱讀容易度檢視。 [上文讀出與原始留出](../semantic_tracking/arrau-bridge-results.md) · [上文讀出封存](../results/arrau-bridge-l4/reference-release.json)

上文候選以來自 ARRAU 的人工指定片語驗證。PEAR 前五和首選命中率分別為 34.4% 和 9.5%，因此介面呈現候選次序。正文中疑問何時出現和解除另須直接觀測。

新介面已用同一個 Qwen3.5-2B 候選在 L4 做實際 Gradio HTTP 驗收。含重複片語、特殊字元和三段正文的輸入只需貼文便取得句級熱圖。接著查兩處同名片語並切換回詞級熱圖，原文和反查座標均一致，所有接續操作共增加零次模型前向。823 筆 ARRAU 可評案例的候選順序和分數與封存評分逐值相同。桌面 1,280 px 與窄螢幕 390 px 的瀏覽器檢查使用合成狀態。 [L4 路徑驗證](../results/gradio-reference-l4/verification.json) · [讀出重播與瀏覽器驗證](../results/gradio-reference-l4/local-verification.json) · [釋放紀錄](../results/gradio-reference-l4/run-verification.json)

同一候選的逐詞標色已通過 MedReadMe 英文留出定位測試。介面分數的句內平均 AP 為 0.713，詞長與反頻率對照分別為 0.504 和 0.565。資料涵蓋受訓人員圈選的難詞與術語。 [留出範圍與 L4 結果](../stage6/medreadme-lexical-results.md)

兩批 IIRC 證據抵達留出共用一份獨立擬合的區間讀出。官方區間 NLL 增益及同文配對辨別率均未達驗收門檻。介面的「提問指向候選」呈現段末的來源片語排名。 [證據抵達留出結果](../iirc-linked-shared-head-results.md)

逐句明細另可查看實驗性評分分布。這項讀出估計群體給 1–5 每個等級的比例，使用加入改寫效應損失的權重。其證據來自 PLABA 開發折外比較。句級熱圖沿用原先的平均分讀出。 [分布評估](../stage-6-holdout-plan.md#加入改寫效應的開發對照)

預設候選使用真正的 Qwen3.5-2B 通過 L4 Gradio 請求驗證。首次分析做一次模型前向，接續三種選取或切換操作皆重用結果。替換後段後，共有前文的 128 個詞仍對齊，最大分數差為 0.00000043 分。IIRC 的 430 段候選排序全部和原 L4 實驗相同。七句測試材料的五級機率與原封存 L4 狀態計算值，最大差為 0.00000024。這些數值是執行路徑的重現誤差，新的真人效度由各留出實驗檢驗。 [L4 驗證](../results/gradio-candidate-l4/verification.json) · [片語留出評分](../results/gradio-candidate-l4/iirc-locked-scores.json)

另以一篇四段、2,496 字元、479 token 的測試文章驗證實際 Gradio HTTP 路徑。只提交正文就會取得全文句級熱圖，同一請求工作階段可切換詞級熱圖。兩種標色都逐字保留原文與段落，26 句與 425 個詞的原文座標和句末或詞末 token 均核對一致。另有一個詞無法精確對齊，原文仍完整顯示。分析用一次模型前向，切換用快取且未增加前向。 [長文驗收程式](verify_article_route.py) · [L4 結果](../results/gradio-article-route-l4/verification.json) · [執行與釋放收據](../results/gradio-article-route-l4/run-verification.json)

用 `python -m research.gradio_app.verify_candidate_evidence` 核對候選模型的權重、實際 Gradio 路徑與兩項真人留出預測的雜湊。此命令只讀取封存檔，無需載入整個語言模型。 [同一候選的驗證清單](../results/gradio-candidate-l4/candidate-evidence.json) · [真人留出結果](../stage-6-holdout-plan.md#同一-gradio-候選的英文留出結果)

在已備妥 CUDA PyTorch 的 NVIDIA L4 上，以 `--cuda` 和同一模型快照路徑啟動。需安裝相同版本的 Gradio、NLTK、NumPy 和 Transformers。

## 核對實際模型與介面

新版全文圖例已用 CLEAR 驗證批次的同一 Qwen3.5-2B 程序做 Gradio 實際 HTTP 驗收。兩篇正文各做一次前向，八次查詢與檢視操作共做零次。另用同一候選的原始句級分數做獨立篇章留出。將分數加入表面特徵後，英文說明文篇章排序的增益區間高於零，文學則跨零。這是整篇文字的順位證據。 [Colab 實際標色](../results/clear-reading-ease-l4/preview.html) · [CLEAR 留出結果](../stage6/clear-reading-ease-results.md)

`verify_overview_regressions` 使用候選的 tokenizer、分句參數和讀出權重檢查段尾縮寫、原文保留、分數圖例，以及從「上文指向」重新分析後的檢視。測試用合成因果狀態執行 Gradio 回呼，無需載入 Qwen。

同一候選也在 L4 通過新版 Gradio HTTP 路徑。先分析一篇文章並選取上文片語，再提交一篇含段尾縮寫、CRLF、表情符號及組合字元的新文章。伺服器回傳閱讀容易度檢視，清除上一份的查詢和選取，並重畫兩個句子標記與 18 個詞彙標記。已核對原文和座標。兩篇各做一次前向，包含查詢與檢視切換的八次後續操作共做零次。預覽頁保存第二篇的實際標色結果。 [標色預覽](../results/gradio-overview-http-l4/preview.html) · [HTTP 驗證](../results/gradio-overview-http-l4/verification.json) · [執行與釋放紀錄](../results/gradio-overview-http-l4/run-verification.json)

```sh
.venv-cogllm/bin/python -m research.gradio_app.verify_overview_regressions \
  --checkpoint research/results/gradio-candidate-checkpoint \
  --tokenizer /path/to/15852e8c16360a2fea060d615a32b45270f8a8fc/tokenizer.json
```

用 PLABA-EVAL 的 `text_01` 重現分數。七句、兩個量尺的 14 個分數與原研究程式逐值一致。本機 Gradio 6.17.3 API 的一次分析請求觸發一次 Qwen 前向，回傳兩種量尺、原文標記和選取句的位置。這項檢查用一篇開發資料核對實作與介面，泛化證據來自先前以文件配對為單位的留出評估。 [研究基線的留出結果](../training-protocol.md#已完成的首輪基線)

```sh
.venv-cogllm/bin/python -m research.gradio_app.verify \
  --model /path/to/15852e8c16360a2fea060d615a32b45270f8a8fc \
  --sentences /path/to/plaba-eval/sentences.csv
```
