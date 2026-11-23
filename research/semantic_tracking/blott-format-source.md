# Blott 等人（2023）的英文句末詞義重解

〔文獻支持〕實驗一的全文顯示條件可檢驗同一讀者遇到兩個相配英文句子時，句末「說得通」的回答是否改變。研究者把線索放在最後一個詞，使有歧義的版本必須採用較罕見的詞義才會說得通。對照版沿用句框，換成符合語境的無歧義主名詞與末詞。研究者安排兩版在不同實驗區塊出現，但對該讀者保持同一呈現方式。這項研究的成對單位是同一讀者對同一句框的兩個版本各作答一次。 [原論文，Experiment 1 §§Materials、Design](https://doi.org/10.1080/23273798.2022.2159992) · [全文與 Table 3](https://madoc.bib.uni-mannheim.de/63934/1/Presentation%20format%20affects%20the%20behavioural%20and%20neural%20processing%20costs%20of%20sentence%20reinterpretation.pdf)

## 刺激、真人回報與時間

全文條件一次顯示整句 4.5 秒，之後顯示 `Sense` 和 `Nonsense` 按鈕。作者保留 108 位使用英式英語的讀者。每位讀者在整句條件下讀 21 對可連貫的句子，以及 18 個不連貫填充句。63 對可連貫的實驗句依讀者分配到整句文字、逐詞文字或語音。本文只取整句文字的可連貫試次。Table 3 的整句條件平均正確率是有歧義 80%、無歧義 95%。兩種連貫版本都以 `Sense` 為正確答案。 [原論文，Experiment 1 §§Participants、Design、Procedure、Results](https://doi.org/10.1080/23273798.2022.2159992)

原表 `Exp1_TaskData.csv` 的 `Item` 和 `Ambiguity` 連到材料表的句框及版本。稿件材料表包含兩個實驗的 90 組句框，第一個表的 63 組連貫版本屬於此比較。126 個 `Sentence` 值皆由單一 `<h4>` 包起。取出實際顯示的正文後，沒有 HTML 實體需要轉碼，所有詞序都與材料表一致。每句可建立末詞線索前最後一個非空白字元、線索結束與句末三個停點。同一對恰好只更換兩詞、且第二處是線索時，也保留較早換詞的主名詞座標。真人只在整句之後做一次選擇，因此本端點沒有讀者看到末詞前的詞義判斷、改變解讀的時機或回看的位置。 [材料表](https://osf.io/download/h53ed/) · [畫面事件與作答欄](https://osf.io/download/trf8w/)

以 `Modality=Vis`、`Plausibility=Plaus`、實驗區塊的 `Screen.Name=Response` 取樣。原始程式列出 16 位應排除的參與者，並用嚴格的 `Reaction.Time > 250.00`。因此恰好 250 ms 的反應也排除。排除 15 筆反應時間及 19 筆其餘逾時後，63 組材料共有 4,502 筆明示作答。公開繪圖 CSV 的 108 位讀者整句平均為 79.85% 與 94.95%。從本文的明示試次分別重算讀者平均，得到 79.53% 與 94.99%，顯示兩份公開檔重算分母後仍有差異。下節的配對效應以實際逐題作答為準。 [作者分析程式，第 43–88 行](https://osf.io/download/uf5pr/) · [公開繪圖彙總](https://osf.io/download/bw4ry/) · [重計統計](../results/blott-format-source-audit.json)

## 來源與既有訓練材料

2021 年 Blott 試次已用於 CogLLM 的英文句末驗收。兩篇研究沿用相關先導方法。192 個舊版本（48 個句框，含 `bridge` 與填充版本）與 126 個新版本之間，沒有正規化後完全相同的句子，也沒有遮蔽主名詞及線索後仍相同的句框。19 組新材料的主名詞和末詞曾在同一個舊版本的主名詞及後續線索範圍一起出現，因此全組剔除。25 組原始新材料與既有材料有至少一段五詞相同，作為查核資訊。 [2021 既有驗收](../stage6/blott-coherence-holdout.md) · [2023 Materials 與 Supplementary Materials](https://doi.org/10.1080/23273798.2022.2159992) · [重疊規則與清單](../results/blott-format-source-audit.json)

〔本研究發現〕保留 44 組句框、88 句、3,139 筆明示作答。其中 108 位讀者有 **1,558 組**針對同一個句框兩版皆有效的回答。每組先做有歧義減無歧義，再將同一句框內的讀者差值平均、對句框等權平均。有歧義版回答「說得通」的比率低 **16.33 個百分點**。88 句均有末詞和句末的原文座標，兩個版本的主名詞位置保留為缺值。 [獨立核對的計數、效應及雜湊](../results/blott-format-source-audit.json)

OSF 明列 CC BY 4.0。整理程式以串流讀取已固定 SHA-256 的原始檔，只在記憶體使用原始讀者識別欄位來套用作者排除名單及產生本機別名。保留句子 ID、別名、試次序位、反應與毫秒時間供評分。模型輸入只存文字、文字 ID 與停點。 [整理程式](blott_format_prepare.py) · [彙總與來源雜湊](../results/blott-format-source-audit.json) · [OSF 授權欄](https://api.osf.io/v2/nodes/m87vg/) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
