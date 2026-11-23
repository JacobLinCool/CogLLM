# 24 組真人資料的觀測模型實作與驗收

新版聯合訓練已接通 24 組來源、62 個觀測區塊。完整開發封包有 27,167 份輸入、164,220 個 Qwen 停點。每組研究的資料表均連到原始研究與發布材料，記錄觀測發生的位置或區間、有效分母、採用的似然，以及相關的共享投影。[來源一至十二](observation-contracts-a.md) · [來源十三至二十四](observation-contracts-b.md) · [程式登記](observation_registry.py) · [完整封包驗證](../results/union-observations-l4/geometry-verification.json)

PLABA 的句級序位評分同時讀被評句末和可見全文末。Frick 的原始按鍵拆成 1,800 筆首次回報或設限路徑，以及 2,866 筆首次回報之後的當句判斷。OneStopQA 與 RACE 的答對率由選項分布和正解索引導出。Rodger 從同一材料的兩個停止位置共同擬合讀者前後評分，再由兩個預測讀出變化。相同資料衍生的 36 個舊版開發效應，已確認可以由新版輸出重算並用於診斷。[聯合訓練規格](observation-protocol.md) · [效應診斷程式](observation_effect_diagnostics.py)

## 獨立驗證

完整封包的 65 個檔案可以從來源重建到逐位元一致。CPU 以合成表徵跑過完整觀測資料，獨立重算 169,025 個預測列及 57 個有開發列的區塊。梯度測試覆蓋全部 62 個區塊，確認只有登記的共享投影與量測頭接到梯度。另核對 Frick 的首事件機率只依已到達的停點，以及 PLABA 回顧評分實際使用兩個登記的停點。[CPU 驗證](../results/union-observations-l4/cpu-smoke-verification.json) · [梯度路由驗證](../results/union-observations-l4/shared-routing-verification.json)

正式全量 Colab 試跑的工作階段在表徵擷取時消失，未回收可驗證的權重，並已確認該 L4 釋放。隨後用確定性抽樣保留 24 組來源、62 個區塊及各自的訓練與開發角色。該封包包含 939 筆完整觀測列、688 份輸入和 1,074 個停點。第二張 L4 成功執行一次 Qwen 因果前向與 12 輪讀出訓練，共 192 次更新。12 次前綴檢查通過，獨立程式重算 57 個開發區塊，結果檔雜湊與硬體釋放均已核對。此抽樣的分數只供整條執行路徑驗收。[全量運算嘗試](../results/union-observations-l4/full-run-attempt.json) · [真實 Qwen 短跑驗證](../results/union-observations-l4/l4-smoke/model-verification.json) · [釋放與結果收據](../results/union-observations-l4/l4-smoke/run-verification.json)

本輪端點在原始量測的詞、句、呈現停點或成對條件上計算。已支持多種觀測共同更新相關投影，正式整組留出仍未評分。Frick 記錄首次回報矛盾。文章中同一個自發疑問的逐 token 解除仍需要對應的真人連續觀測。
