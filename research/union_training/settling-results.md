# 全量聯合訓練的續訓與整組留出驗收

以凍結的 Qwen3.5-2B 完成全量觀測聯合續訓，共用 24 組公開真人資料、62 個觀測區塊及 164,220 個因果停點。第 55 輪達到事先固定的開發損失停訓條件，第 31 輪驗證分數最低，選為唯一正式留出候選。[續訓與留出規格](settling-and-heldout.md) · [候選權重](../results/union-observations-settled-l4/heads.npz) · [權重收據](../results/union-observations-settled-l4/seal.json)

## 開發集停訓與重播

由第 12 輪權重重新開始 AdamW 動量，續訓 43 輪、6,106 次更新。第 12 輪的來源等權開發 NLL 在 L4 重播為 1.19914，第 31 輪為 1.17078，降低 0.02836。第 39 與 47 輪評估後依規格降學習率。最後八輪沒有達到 0.001 的改善門檻，第 55 輪停訓。第 55 輪分數為 1.20351，正式候選保留第 31 輪的最佳權重。這是固定開發集與學習率日程下的停訓判定。[封存規則](../results/union-observations-settled-l4/policy.json) · [完整歷程](../results/union-observations-settled-l4/fitting.json)

獨立程式對 169,025 筆訓練及開發預測重算 57 個開發區塊，另驗證兩次降學習率與停訓決策。和第 12 輪相比，52 個區塊的開發 NLL 下降、5 個上升。CompLex 詞彙難度的 NLL 從 1.1730 上升到 1.5985。後續位置評估把兩項 Frick 配對效果從錯置的第 13 句修到原研究的第 14 句。修正後，36 個開發配對效應的平均標準化絕對誤差從 0.11943 降至 0.11827。 [獨立重算](../results/union-observations-settled-l4/model-verification.json) · [原配對封存](../results/union-observations-settled-l4/development-effects.json) · [Frick 位置修正](localization-results.md#首次察覺矛盾的句間時序)

續訓 L4 還原了已封存的全部 18 批凍結表徵，新增擷取 0 批。配置至釋放間隔約 13 分 5 秒。報告使用工作階段時間戳計算用時。兩批已封存歷程的擬合計時欄位曾被更新迴圈的索引覆寫，已修正程式以分開記錄擬合開始時間。[工作階段與釋放收據](../results/union-observations-settled-l4/run-verification.json) · [批次接續驗證](observation-full-results.md)

## 整組真人留出

先固定第 31 輪權重與各研究的預測規格，再對 164 份留出刺激及 216 份 Paape 校準刺激各做一次因果前向。三組材料各有 12 項因果前綴檢查。正式反應留在本機，各研究的預測封存後才評分。這張 L4 已核對釋放。[材料、停點及釋放收據](../results/union-observations-formal-l4/extraction-verification.json)

| 留出觀測 | 結果 |
| --- | --- |
| SAP 的消歧詞自定步速讀時 | 三種句法構式有一種的消歧詞效應方向符合真人。納入下一詞及再下一詞後，九個位置對照有六個方向一致。NP/Z 在消歧詞和下一詞均預測為反方向。 [結果與逐位置表](formal-sap-spr-results.md) |
| Anglin 的首日信念更新 | 六項同人變化有三項方向一致，零項通過 ±0.5 分的幅度容差。兩項支持證據後，真人平均增加 1.043 分，模型估計增加 0.0125 分。 [結果與量尺](formal-anglin-results.md) |
| 繁中 MECO 的篇後理解 | 1,440 次真人作答的實測答對率為 81.32%，篇末讀出的平均預測為 56.12%。模型 NLL 為 0.62165，固定來源基線為 0.48207，差 +0.13958 nats／試次，95% 區間 [0.07042, 0.20626]。 [結果與對位更正](formal-meco-results.md) |
| 同一 SAP 家族的 Timkey 眼動 | 四項主比較有三項方向一致。四項皆低於詞長與詞序基線的表現，配對 NLL 差的 95% 區間均高於零。首次離開時，真人歧義減控制的回視率為 +19.10 個百分點，模型估計為 −0.24。 [結果與原量尺](formal-timkey-eye-results.md) |

Timkey 讀出只以英文 OneStop 同類眼動量測擬合，沿用固定第 31 輪 `processing` 狀態及既有 SAP 的 288 個詞末表徵。額外 L4 對 339 份 OneStop 正文各做一次前向，擷取 36,983 個停點，並已確認釋放。獨立位置查核逐字核對兩研究使用的同一句與詞末。 [逐詞位置對齊](../results/union-observations-formal-l4/timkey/timkey-roi-alignment-independent.json) · [擷取與釋放收據](../results/union-observations-formal-l4/timkey/extraction-verification.json)

第 31 輪候選在指定整組留出上仍無法一致重現真人反應。MECO 是篇章層級的群體答對率，Anglin 評估文字轉錄後的首日信念，SAP 的兩種閱讀方式分別評估消歧處及後續詞的條件讀時與眼動。Anglin 的確定程度及隔日反應、SAP 的句末解讀尚缺相符的讀出，留為未評。這批結果尚不足以驗證任意文章的逐 token 認知負荷曲線。
