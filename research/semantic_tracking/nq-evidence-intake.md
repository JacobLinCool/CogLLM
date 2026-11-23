# Natural Questions 留出資料的取得與核對

從已排除 IIRC 來源重合的官方 validation 索引，以 `SHA256('cogllm-nq-v1:' + revision)` 排序頁面版本，取前 72 組及其全部 153 筆原始題目。索引 SHA-256 為 `6a4a3921b7fa142eaf34db148e0f93a44fd0da6313a2e9c2b63b8853cad247b5`。下載前固定選取規則與完整 ID。資料來自 Hugging Face 的 `google-research-datasets/natural_questions`，使用 `default/validation` 與 commit `e8103d566bef4154c2c12b17c6095ec5275840cc`。 [發布資料](https://huggingface.co/datasets/google-research-datasets/natural_questions) · [Google 原始資料與授權](https://ai.google.com/research/NaturalQuestions/download)

本機原檔目錄為 `.cache/cogllm/natural-questions/holdout-raw-v1`。`grade-free.jsonl` 保存原索引、ID、題目、完整原始 HTML、token 欄位及作者提供的可選區塊。`outcomes.jsonl` 以 `0600` 保存相同 ID 的原始 `annotations`。預測封存前只驗證記錄識別與檔案完整性，人工標註留待評分。

| 檔案 | SHA-256 |
| --- | --- |
| `manifest.json` | `70396f1836192abfceec936de5612c583639704bdba9f2a16b4adb2dc3f840e3` |
| `grade-free.jsonl` | `1c35576a14366846ff6ebbb11fb86def078cec4121612aec8a66341e0ddaf698` |
| `outcomes.jsonl` | `f1b87047a2fef50293dbd037a68ae59cb5e4e59608d563affa9afed27ab2333e` |

交付的本機檔由兩個取得路徑組合。Hugging Face Rows API 先完成 100 筆，驗證 `X-Revision`、原索引、ID、URL 與截斷欄位後，以每四筆一組保存雜湊。服務限流後，剩餘 53 筆使用上述 commit 的 `default/validation-00000-of-00007.parquet` 至 `default/validation-00006-of-00007.parquet`。逐分片串流寫入暫存磁碟檔，檢查與發布清單相同的分片長度。以 `pyarrow.parquet.ParquetFile(path, memory_map=False, pre_buffer=False, buffer_size=262144).iter_batches(batch_size=4, use_threads=False)` 讀取，依索引的 `row_in_shard` 選取對應列才轉 Python。核對 ID、URL 與欄位後，用同一個 JSONL 寫入函式分離各列，最後依固定選取順序合併。暫存分片逐一移除。成功保存的 API 內容為 13,058,839 個壓縮位元組，Parquet 分片合計 1,338,488,354 位元組。Parquet 程序實測 RSS 峰值 742.8 MiB，程序已結束。

限流解除後，Rows API 獨立完成全部 153 筆。成功保存的壓縮內容為 18,789,577 位元組。完整 Rows 結果與上表兩個資料檔均逐位元相同。下列命令用獨立輸出重建相同資料檔。收據會記錄這次取得路徑及傳輸量。 [取得程式](nq_evidence_fetch.py) · [Rows API 文件](https://huggingface.co/docs/dataset-viewer/en/rows)

```sh
python3 -m research.semantic_tracking.nq_evidence_fetch \
  --index .cache/cogllm/natural-questions/iirc-disjoint-index.jsonl \
  --output .cache/cogllm/natural-questions/holdout-raw-v1-replay
```
