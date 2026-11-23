# MedReadMe 難詞與片語的原文對齊

作者的標註團隊經訓練後獨立圈選術語與難片語，再由裁決者核對。標註使用明定的七類詞彙，包含一般難詞、縮寫、醫療名詞及兩種查詢難度的醫療術語。這可作為介面逐詞著色的外部定位對照。作者未提供同一批讀者逐詞自評、疑問時間資料或個人票數。 [原文 §§2.1–2.3](https://aclanthology.org/2024.emnlp-main.958.pdf#page=3)

固定作者發布的 `jargon.json` 和 `readability.csv`。在每個來源、原文或簡化版本及切分內，以只移除空白後相同的字元序列做唯一配對。逐段核對標註 token 座標、原始片段及映射回 CSV 原句的字元範圍。發布的測試切分有 1,140 句，1,131 句可唯一對上，共保留 2,130 個標註區段。另有 9 句無法消歧，未納入對齊。 [固定版本的發布檔](https://github.com/chaojiang06/medreadme/tree/7c52205d18f16b80e3ffccb4e9da88b1942df310/dataset) · [對齊稽核](../results/medreadme-span-audit.json)

在 194 個難以經短時間查詢理解的醫療術語中，145 個包含至少兩個 Gradio 可著色的完整詞，49 個包含一個。將現有逐詞讀出對上這些標註，可分別測單詞定位及多詞區段覆蓋。資料未發布文章 ID、簡化配對 ID 或逐人評分。正式測試須先核對與擬合資料的文字重合，並標明評估的分組粒度。CEFR 句級評分另有不同施測流程，原文對齊只建立區段標註的可用範圍。 [術語規則與標註限制](https://aclanthology.org/2024.emnlp-main.958.pdf#page=7) · [既有句級稽核](textcomplexityde-transfer.md#其他資料的可用範圍)

```sh
python -m research.stage6.medreadme_span_audit \
  --raw .cache/cogllm/medreadme-audit/raw \
  --output research/results/medreadme-span-audit.json
```

兩次核對產生相同結果。此階段只釋出彙總計數與重現程式。 [程式](medreadme_span_audit.py)
