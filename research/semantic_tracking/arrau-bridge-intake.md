# ARRAU 2.1 橋接先行詞留出資料

公開版 GNOME 收錄 5 篇醫療與藝術史文件，PEAR 收錄 20 篇口述故事逐字稿。兩個集合保留作者發布的全量文件，來源未附訓練、驗證或測試切分。程式驗證兩份壓縮檔與 Apache-2.0 授權檔的 SHA-256，版本固定在 `764d5e53c10656ff1858503c1d32752b03241174`。 [官方資料與授權](https://github.com/arrauproject/data/tree/764d5e53c10656ff1858503c1d32752b03241174) · [標註論文](https://doi.org/10.1017/S1351324919000056)

| 集合 | 文件 | MMAX 詞位 | 橋接連結 | 可評單一先行詞 | 兩端各位於不同句 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GNOME | 5 | 21,458 | 693 | 549 | 400 |
| PEAR | 20 | 14,059 | 333 | 306 | 261 |

讀取 `related_object=yes` 的主、次欄位，並將 `related_phrase` 指向的標記解析為確切詞跨度。GNOME 的 693 條來源連結含 1 條次欄位連結。主評分只保留一個完整先行詞跨度，該跨度必須完全位於觸發跨度之前。缺少先行詞、需要多個標記、非連續跨度與位置重疊的註解留在本機查核記錄。跨句評分要求兩個跨度各自完全落入原始 MMAX 標註的不同句段。GNOME 僅有 5 個文件抽樣單位，PEAR 的句段屬口述逐字稿。這兩份資料的觀測是人工語篇標註，未測量讀者即時困惑或理解解除。 [資料建置程式](arrau_bridge_prepare.py)

`model_input` 每篇只輸出 `id`、`text`、`stops` 與 `paragraphs`。`text` 保留逐詞 MMAX 字串，以單一空格分詞，以換行保留原有句界。`stops` 包含全部標註詞位終點。原始 RAW 文字即使去除空白也無法逐字重現 MMAX 詞序，因此所有 `paragraphs` 均為空。兩份資料均可測詞位和跨句先行詞定位。答案與原始標記僅在 `scoring` 和 `private_metadata`。

使用 Unicode NFKC 正規化、大小寫摺疊及去除標點後，逐篇比對 GUM 的三個切分與 GENTLE。GNOME 和 PEAR 均未發現全文相同，或與任何一篇對照文件共享連續 12 個詞。這項驗證涵蓋本專案現有留出和擬合來源，並未查閱基礎模型的預訓練語料。

```bash
python3 -m research.semantic_tracking.arrau_bridge_prepare \
  --source .cache/cogllm/arrau-bridge/source \
  --output .cache/cogllm/arrau-bridge/prepared
```

`audit.json` 記下每份輸出雜湊。兩次建置的所有輸出逐位元相同，且重跑會拒絕覆蓋已存在的輸出。逐筆驗證確認所有先行詞與觸發跨度都對應合法停點，跨句列的兩端均落在所聲明的句段中。此步驟沒有擬合模型或評分。

## Qwen 前綴座標驗證

使用現行候選的固定 tokenizer，逐一比對截到來源詞末時的獨立編碼與完整文本的 token 前綴。25 篇的 35,517 個詞末全部一致，最長文件為 6,816 個模型 token。855 筆橋接標註的兩端皆達到分詞條件，其中 661 筆跨句。前置工作只用了 tokenizer，單執行緒 CPU 耗時 39.6 秒，峰值約 259 MiB。實際排名的候選條件與結果另有紀錄。 [前置收據](../results/arrau-bridge-preflight.json) · [不含答案的封包建置程式](arrau_projection_prepare.py) · [留出排名結果](arrau-bridge-results.md)
