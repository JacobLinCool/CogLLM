# Gradio 逐詞標色的 MedReadMe 留出結果

使用現行 Gradio 候選 checkpoint 的同一份 CompLex 詞末權重，在單張 L4 對 1,109 個留出原句各執行一次前向。只將原句送到模型端。評分採介面顯示的兩位小數及原文字元座標。可評詞共 23,286 個，精確對齊 23,284 個，覆蓋率 99.991%。其中 666 句同時有標註及未標註詞，可做句內排序比較，涵蓋 15 個發布來源。 [事先封存的規格](medreadme-lexical-holdout.md) · [L4 及釋放收據](../results/medreadme-lexical-l4/run-verification.json) · [實際評分](../results/medreadme-lexical-l4/scores.json)

| 詞序對照 | 句內平均 AP | 候選減對照的 95% 配對區間 |
| --- | ---: | ---: |
| Gradio 詞末讀出 | 0.713 |  |
| 詞長 | 0.504 | [+0.125, +0.289] |
| CompLex 擬合語料的反頻率 | 0.565 | [+0.091, +0.206] |

候選通過預定的覆蓋及排序門檻。結果適用於 MedReadMe 英文單句中，受訓標註員圈選的難詞與醫療術語。同一個模型分數目前可用於排序，個人閱讀中的負荷強度及連續篇章的詞級效度仍需對應的真人資料。 [同一 Gradio 候選的驗收收據](../results/gradio-candidate-l4/candidate-evidence.json)
