# 下一句判為一致，能否代表原疑問已解除？

Frick 與 Schüler 的逐句任務要求讀者判斷**當前句**是否與前文矛盾。讀者在目標句首次按矛盾鍵 `D` 後，兩個條件都超過 92% 在下一句按一致鍵 `K`。這個轉換發生在另一句的判斷上。若直接把 `D → K` 當作同一疑問解除，模型會取得很高的表面解除率，卻缺少對原矛盾的確認。 [原研究 Experiment 1](https://doi.org/10.3758/s13421-026-01900-7) · [資料與對齊稽核](frick-consistency-source.md)

| 文字條件 | 全部試次 | 目標句首次 `D` 且有下一句 | 下一句 `K` | 中間按 `K` 後，較晚的句子又按 `D` |
| --- | ---: | ---: | ---: | ---: |
| `01` 一致 | 900 | 28 | 26／28（92.86%） | 1／26 |
| `02` 不一致 | 900 | 458 | 424／458（92.58%） | 5／424 |

從 180 位讀者的 1,800 條純文字軌跡重算。先排除第一句的指定按鍵位置，再取目標句是 `D` 且前面已評句皆為 `K` 的試次。下一句 `K` 的分母只含有下一句回報的試次。最後一欄只看下一句按 `K` 且其後仍有回報的試次。20 對故事的目標句後文字及句序均一致。表中比例條件在各版實際出現的首次 `D`，兩組分母是不同的觀測集合。研究每次詢問新句是否矛盾，後續的 `D` 也只能定位另一個當句回報。這些計數可檢查解除標籤的定義，解除機率本身仍需要對同一疑問的後續評分。 [試次端點](frick-consistency-protocol.md) · [本次計數與來源雜湊](../results/frick-resolution-negative-control.json)

## 可直接核對的同疑問資料

TED-Q 已將原提問者、疑問和後續評分連在一起。每兩句一個停點，可估計同一疑問第一次達到「至少部分獲答」的區間。未達門檻便結束追蹤時按實際觀測截止處理。達門檻後停止追蹤，完整解答需要另外的觀測。可先固定按演講留出，檢查同一可見文字下，正確疑問相對換入的其他疑問是否改善首次達標區間的預測。 [TED-Q §§3–4](https://aclanthology.org/2020.lrec-1.141.pdf#page=3) · [六篇演講與逐次檔](https://github.com/amore-upf/ted-q) · [序位與截止規則稽核](../stage-5-semantic-run.md)

TED 條款 §§6.3–6.4 將逐字稿用於模型評估列入另需 TED 書面授權的用途。若採這個來源，向 TED Licensing & Distribution 的表單明定 TED-Q 六篇已發布逐字稿及停點片段、Qwen3.5-2B 的學術評估、是否會在 Colab 處理，以及所需的快取期間及公開結果範圍。原始逐字稿和個別作答的再發布及讀出訓練各需依核准範圍辦理。 [TED 使用條款 §§6.3–6.4](https://www.ted.com/about/our-organization/our-policies-terms/ted-com-terms-of-use#h--copyright-issues-and-licenses) · [TED 授權說明](https://help.ted.com/hc/en-us/articles/360004233294-How-do-I-license-TED-or-TEDx-content) · [TED 授權表單](https://media-requests.ted.com/)
