# Natural Questions 的獨立證據區間資料查核

Natural Questions 的開發集保留同一題的五份獨立人工參考。每份參考選取含充分證據的 HTML 區段，或選擇沒有足夠證據。區段同時以來源 token 和 UTF-8 byte 位置記錄。包含不同問題的同一 Wikipedia 版本，可用以檢查模型在同一前綴有沒有分辨不同的證據。資料依官方發布的 CC BY-SA 3.0 使用。 [作者發布與授權](https://ai.google.com/research/NaturalQuestions/download) · [原始量測與標註](https://aclanthology.org/Q19-1026/) · [Hugging Face 檔案](https://huggingface.co/datasets/google-research-datasets/natural_questions)

固定 Hugging Face 版本 `e8103d566bef4154c2c12b17c6095ec5275840cc` 的七個 validation shard。逐檔查詢必要欄位後得到 7,830 題、7,369 個 Wikipedia 原始版本。有 365 個版本包含至少兩個不同的正規化題目，共有 627 個同文題目配對。資料可行性查核發現 352 對的兩題各有至少兩份非空長答案，其中 112 對的全部非空長答案區間按原始 token 座標嚴格分開。選取推論材料時使用獨立的無標註索引，後兩項統計只說明能否檢查配對順位。

為避免和已用於 CogLLM 的 IIRC 頁面重合，將 Wikipedia 標題與 IIRC 訓練及開發的原文標題和可見連結標題做正規化比對。365 個多題版本中，5 個匹配原文、90 個匹配連結，其中有交集。排除後保留 273 個版本、610 筆題目紀錄，正文取回時仍需核對每個版本各紀錄的實際文本。抽查的第一個多題版本中，兩題的完整 HTML 和 7,962 個來源 token 逐值相同。索引重建逐位元一致。 [索引盤點](../results/nq-evidence-index.json) · [選取程式](nq_evidence_index.py)

已在四筆不含人工選取區段的原始樣本中，核對 31,225 個來源 token 和 507 個可能的支援區段。模型用的文字依作者的非 HTML token 順序生成，座標程式逐區段驗證 byte 起訖，並從同一批 token 還原本文的 Unicode 字元範圍。原始 HTML 及人工答案不需要進模型。表格及清單攤平成文字後的結果需獨立報告。 [資料格式和長答案的類別](https://github.com/google-research-datasets/natural-questions#long-answer-candidates) · [區段對齊程式](nq_source_positions.py)

此資料衡量看到多少文字後足以回答已知問題。對同一問題，以五份等權參考計算區段誤差。真人未選出完整支援區段時，用觀測窗之後的剩餘機率計分。以版本識別碼及實際可見 token 確認同一正文，題目和正文擷取後先封存預測，再以參考區段評分。已固定的 ExpMRC 四組參數可直接轉移，以判斷其機率校準能否跨語料。真人從正文何時自發提出疑問及何時意識到答案，仍需另一種觀測。 [已封存的參數與公平對照](evidence-calibration-comparison-results.md)
