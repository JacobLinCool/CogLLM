# 全量聯合訓練與 L4 接續驗證

全量聯合訓練已完成。使用 24 組真人資料的 62 個觀測區塊、27,167 份輸入和 164,220 個停點。18 批凍結 Qwen3.5-2B 表徵均保存至本機。訓練前依原始順序合併，核對每個文件和停點只出現一次。三張 Colab L4 均確認釋放。[分批與接續規格](observation-resume.md) · [全量運算收據](../results/union-observations-full-l4/run-verification.json)

## 實際接續

第一張 L4 傳回第 1 批後，控制器用錯誤的清單型別呼叫封包驗證器，因此提前釋放該工作階段。修正後直接用已下載的 6 個分片還原第 1 批，並核對其 9,531 個停點。第二張 L4 擷取第 2 至 15 批，於封存第 16 批時失聯。控制器確認第二張 L4 已釋放後，自動配置第三張，只擷取第 16 至 18 批。前 15 批從本機回填後才合併並訓練。對完成後的執行目錄再次執行接續命令，直接驗證已封存的結果。18 個最終批次全部通過 SHA-256 核對；206 項前綴檢查最大相對 RMS 差為 0.00000870，門檻為 0.001。[實際批次與回填明細](../results/union-observations-full-l4/recovery-verification.json) · [前綴檢查](../results/union-observations-full-l4/causality.json)

## 聯合訓練

凍結骨幹，共訓練 12 輪、1,704 次讀出更新。每輪使用 135,819 筆訓練觀測列。獨立驗證程式核對 169,025 筆訓練及開發預測，重算 57 個開發區塊的損失和 36 項既有配對效應。第 1 輪至第 12 輪的來源等權開發 NLL 從 1.32764 降至 1.19914，越低表示模型對實際觀測給予較高的平均概似。最佳值落在預定預算的最後一輪，尚未建立收斂證據。後續重新對齊兩項 Frick 目標句效果後，36 項效應的平均標準化絕對誤差為 0.11943。正式整組留出在這一階段仍封存。 [訓練歷程](../results/union-observations-full-l4/fitting.json) · [獨立驗證](../results/union-observations-full-l4/model-verification.json) · [原開發效應封存](../results/union-observations-full-l4/development-effects.json) · [修正後的開發效應](localization-results.md#首次察覺矛盾的句間時序) · [全量讀出權重](../results/union-observations-full-l4/heads.npz)

納入初次中斷與後續回填，三張 L4 的配置至釋放間隔合計約 47 分 17 秒。18 個表徵批次的傳輸封包共約 1.17 GiB。擷取記錄的峰值 CUDA 保留顯存為 10.1 GiB。[配置與釋放紀錄](../results/union-observations-full-l4/run-verification.json) · [擷取收據](../results/union-observations-full-l4/extraction.json)

已從上述第 12 輪繼續訓練，於第 55 輪按開發損失停訓，選出第 31 輪做正式留出驗收。[續訓與正式留出](settling-results.md)
