# QASPER 正文證據到達實驗

在 23 篇留出論文上，加入正文降低了標註證據的段落定位誤差。交換同篇疑問後，分數接近使用原疑問的條件。同一停點上，模型在 41 對疑問中有 21 對判對哪個疑問已讀到標註證據。 [實驗計畫](semantic-joint-first-run.md#下一輪正文證據對照) · [完整留出結果](results/qasper-evidence-first-run/scores.json)

## 資料與評分設計

QASPER 的提問者可見標題及摘要，被要求提出需要讀正文才有答案的疑問。另一批讀者可見整篇，標出足以支持答案的證據，在證據充分的前提下優先選較早段落。首輪只納入沒有事先讀過該論文的提問者，以及可唯一對齊且所有發布標註同意的單段正文證據。按論文保留官方訓練及測試切分，選用 60 篇訓練論文的 151 個疑問及 23 篇測試論文的 55 個疑問。每篇正文維持原有順序，截點先由 14,000 字元、32 個正文段落與 6,000 個原生 token 三個上限決定，再查證據是否落在該範圍。目前結果涵蓋這一類文字證據。 [原研究 §2.2](https://aclanthology.org/2021.naacl-main.365.pdf#page=3) · [QASPER 公開資料](https://huggingface.co/datasets/allenai/qasper) · [建置程式](semantic_tracking/qasper_prepare.py)

固定 Qwen3.5-2B。每篇文章包含標題、摘要及原順序段落，在同一次因果前向擷取段末狀態和兩個停點間的平均狀態。疑問以同一骨幹獨立編碼以供研究讀出，同篇論文的各個疑問共用同一份文章表徵。每個停點的內容特徵只用當下及已讀文字。Qwen 的狀態凍結，訓練具相同 L2 係數的線性讀出。 [因果特徵](semantic_tracking/qasper_extract.py) · [讀出](semantic_tracking/qasper_fit.py)

| 條件 | 可見內容 |
| --- | --- |
| 位置基線 | 正文段落順序、可見字元數與當前區段長度。對同篇同停點的所有疑問給相同的分數。 |
| 正文＋原疑問 | 加入正文狀態及其與原疑問表徵的交互特徵。 |
| 正文＋交換疑問 | 重用同一讀出，保持正文及停點，用同文章的另一個問題的機率評估原疑問的證據區間。這個對照不另擬合，也不重新執行原文的骨幹。 |

正文以段落為觀測單位。已讀完標題及摘要為位置 0，第 \(e\) 段有全部標註證據時，訓練區間為 \((e-1,e]\)。在第 \(t\) 段的條件機率為 \(h_t\)，存續機率為 \(S(t)=\prod_{j=1}^t(1-h_j)\)。取 \(-\log[S(e-1)-S(e)]\)，越低代表模型分配給標註證據區間的機率越高。較早的前綴可能有未被圈選的線索，因此這裡的事件只代表所採用的標註證據抵達。 [區間擬合](semantic_tracking/qasper_fit.py)

全部問題比較正文組和位置基線。三組匹配對照涵蓋同一篇論文的 41 對疑問，且每對的兩個證據段落不同，交換兩個疑問的方向均納入。另在較早證據到達的同一停點檢查模型是否給該問題較高的「已讀到證據」機率，平手算半分。重抽樣的單位為論文，使用同一折預測及同一試次。保留測試證據標註於本機，推論輸入只有未標記的文字、停點及文章與疑問的對應鍵，評分前核對預測封存。 [評分與配對](semantic_tracking/qasper_score.py)

## 留出結果

| 納入同樣的 41 對疑問、82 個計分方向 | 證據區間平均 NLL（nats，越低越好） |
| --- | ---: |
| 位置基線 | 3.2670 |
| 正文＋原疑問 | 3.1061 |
| 正文＋交換疑問 | 3.1294 |

正文加原疑問相對位置基線的 NLL 差為 −0.1610 nats，95% 配對論文區間 [−0.2869, −0.0431]。相對同一正文的交換疑問條件，差為 −0.0234 nats，區間 [−0.0822, +0.0343]。不做疑問配對時，55 個疑問的正文組和位置基線 NLL 分別為 3.0370 與 3.2273，配對差 −0.1904 [−0.3104, −0.0752]。 [分數與配對區間](results/qasper-evidence-first-run/scores.json)

在同一篇的較早證據段末固定停點，位置基線給兩個疑問相同分數。一半計分下為 50%。正文組的已讀到證據累積機率排序為 21 勝、20 負，達 51.2%，比位置基線高 1.2 個百分點，95% 區間 [−11.8, +14.7] 個百分點。只取當前段落的條件機率，命中數同樣是 21 對。正文特徵有助於定位被標註的證據區間，但原疑問與交換疑問的對照不足以確認這個讀出能辨識正在回答哪個疑問。這一輪未達先前所訂的聯合訓練納入條件。 [同停點排序與原先驗收準則](results/qasper-evidence-first-run/scores.json) · [研究計畫](semantic-joint-first-run.md#下一輪正文證據對照)

## 執行與重現

Colab L4 擷取 83 篇論文及 206 個疑問，共 222,272 個非填充 token、2,214 個對齊停點，無分詞對齊排除。12 項前綴正反控制通過。擷取與讀取耗時 95.05 秒，擬合程式內計時 2.03 秒，擷取時 CUDA 記憶體峰值保留 9.45 GiB。配置確認至發出停止要求為 202.41 秒，已確認釋放。帳戶同時有其他活動指派，整體餘額從 760.86 到 760.49 CU。 [特徵擷取](results/qasper-evidence-first-run/extraction.json) · [因果檢查](results/qasper-evidence-first-run/causality.json) · [執行與釋放](results/qasper-evidence-first-run/manifest.json) · [帳戶快照](results/qasper-evidence-first-run/usage.jsonl)

用作者發布的 QASPER v0.3 封存檔重建資料。兩個封存檔的雜湊、所選論文的 arXiv 來源及使用邊界保存在稽核。QASPER 採 CC BY 4.0，底層文本沿用原作者公開的 CC BY 系列篩選，未另外逐篇核對該系列的各別標記。重建檔逐位元一致，全部證據均重新對回原始資料。依序執行 `qasper_prepare`、`qasper_extract`、`qasper_fit` 及本機 `qasper_score`，留出預測經雜湊封存。評分重跑逐位元一致，區間與同停點排序另以獨立算式核對。 [資料稽核](results/qasper-evidence-first-run/data-audit.json) · [建置核對](results/qasper-evidence-first-run/preparation-verification.json) · [讀出權重](results/qasper-evidence-first-run/heads.npz) · [評分驗證](results/qasper-evidence-first-run/independent-verification.json)
