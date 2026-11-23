# 全量聯合訓練的表徵接續

完整開發封包沿用 24 組來源、62 個觀測區塊及既有材料分割。分批器將 27,167 份輸入、164,220 個已核對停點及 2,598,224 個 token 依原始文件順序切成 18 批。每份輸入只屬於一批。批次上限為 150,000 token 和 10,000 個停點，個別文件的前向流程維持完整。[觀測規格](observation-protocol.md) · [批次計畫與合併](observation_shards.py)

## 保存與接續

每個批次帶有原觀測封包、模型版本、tokenizer、擷取程式及研究規格的 SHA-256。L4 擷取程序在同一工作階段載入一次凍結 Qwen。每份輸入做一次正式因果前向，另做獨立前綴核對。批次完成後立即發布表徵、位置、擷取收據及前綴檢查，以 12 MiB 檔案傳回本機。控制器先比對分片及完整封包，再驗證表徵座標和因果檢查。完整檔案通過後才正式保存該批。[遠端執行程式](shard_worker.py) · [本機控制器](checkpointed_colab.py)

原工作階段消失而本機仍有完整批次時，控制器在下一張 L4 只要求尚缺的批次。各批在本機驗證後，訓練工作階段可由其一手接收的表徵與本機恢復的表徵組齊。最終收據記錄所有批次的雜湊。在配置視窗內可由原執行目錄再次呼叫控制器，使用同一份不可變批次計畫。本機模擬驗收覆蓋了提交第一批後斷線、補擷取其餘批次及恢復原批次的流程。

## 聯合訓練驗收

合併依原批次順序以記憶體映射寫出全量表徵，核對每一停點恰好一份。之後使用新版量測似然與共享投影跑 12 輪，並在材料分組開發折選擇檢查點。訓練及結果封存依每批擷取收據綁定。用原始尺度的封存預測重算開發損失和 36 項配對效應。正式整組研究沿用事前封存的角色。[訓練](fit_observations.py) · [結果驗證](verify_observations.py) · [效應診斷](observation_effect_diagnostics.py) · [正式封存](holdout-plan.md)

## 執行入口

在專案根目錄以已安裝 NumPy 和 tokenizers 的 Python 環境建立批次封包。第一個命令輸出執行目錄。

```sh
python -m research.union_training.checkpointed_colab prepare \
  --packet .cache/cogllm/union-training/observations-2 \
  --tokenizer-path "$QWEN_TOKENIZER" \
  --holdout research/union_training/holdout.json \
  --output .cache/cogllm/union-training/checkpointed
```

啟動或繼續同一封包時，將原執行目錄傳給相同命令。控制器核對可以使用的已存批次與已收回的分片，再依剩餘工作配置 L4。結果提交到該目錄的 `fit/`，每一張 L4 的狀態寫在 `sessions/`。

```sh
python -m research.union_training.checkpointed_colab run "$COGLLM_RUN_DIR"
```

全量執行已完成，並在一次遠端工作階段失聯後，從 15 個已驗證批次自動接續。 [全量訓練及實際回填紀錄](observation-full-results.md)
