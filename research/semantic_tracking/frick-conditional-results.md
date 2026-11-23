# 同句配對訓練小幅增加目標句效應

在 Frick 的 20 組故事回溯留出，加入匹配目標句的條件概似後，首次回報的前文條件差從 **17.15 提高至 18.95 個百分點**，增量 +1.80，95% 區間 [+0.77, +2.98]。真人差為 **48.27**。模型減真人的區間 [−40.05, −18.46] 超出 ±10 個百分點的幅度門檻。 [原診斷](frick-target-hazard-diagnostic.md) · [固定規格](frick-conditional-protocol.md) · [評分結果](../results/frick-conditional/results.json)

讀出沿用已有的七個因果特徵，將四類位置交叉熵和目標句兩種前文的條件概似加進整篇首次事件損失。條件概似固定每個訓練配對的兩個風險集人數和首次回報合計，使用目標句兩個 logits 的差。20 折按故事留出，各折訓練配對方向、特徵尺度及內層準則均在該次訓練資料重算。所有折的內層均選位置項權重 4、配對項權重 4。其他句子的觀測用於整篇損失與位置交叉熵。這次使用已評過的材料來診斷學習目標，獨立語料泛化仍需另測。 [模型條件概似](frick_conditional_fit.py) · [各折選擇](../results/frick-conditional/fits.json)

一致與不一致版的目標句前偏差為 +3.65 和 +3.25 個百分點，兩個區間都進入 ±10 的門檻。整篇 NLL 為 1.5939 nats／篇，相對四類位置校準增加 +0.0202 [0.0041, 0.0370]。相對七特徵等權讀出的差為 −0.0508 [−0.1829, +0.0878]，區間上界超過 +0.05 的非劣界限。結果保留為配對訓練的回溯對照。 [分項及比較](../results/frick-conditional/results.json)

封存四份產物逐位元重跑一致。獨立檢查移除留出故事的目標與真人作答資訊後，重建的 584 個 logits 全部相同。用已擷取的同一凍結 Qwen 句末狀態，單執行緒擬合及評分約 1.57 秒，峰值約 111 MiB。 [執行入口](frick_conditional_score.py) · [核對收據](../results/frick-conditional/independent-check.json) · [資源及重現](../results/frick-conditional/run-verification.json)
