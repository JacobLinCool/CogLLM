# Singer 等人的逐句矛盾判斷

Singer、Spear 與 Rodrigo-Tamarit（2023）的實驗二提供 **40 篇英文故事的目標句判斷統計及個別試次正誤**。受試者每看完一個句子，就按鍵判斷它是否與先前內容一致。作者記錄目標句的判斷和時間，後續句只記錄時間。因此這份釋出可作為另一批讀者的目標句效應參照。要評估首次按下矛盾鍵的位置，仍需保存各句的判斷。 [原論文 Experiment 2，Procedure、Table 3 及註 3](https://doi.org/10.3758/s13421-022-01351-w) · [作者 OSF 存檔](https://osf.io/a6xqk/)

## 實際讀到的公開資料

| 作者檔案 | 已核對的資料 |
| --- | --- |
| [目標句正確性 CSV](https://files.us.osf.io/v1/resources/a6xqk/providers/osfstorage/62bb62db9e9e6600fe8a13a3) | OSF v1，45,505 bytes，SHA-256 `b9100913b95f85d842a7bd795718a54664e03aab1fcda64bde02e2a6c43a33db`。欄位為 `Subject, Material_Set, Passage, Truth, List, Accuracy`。共 3,080 筆，涵蓋 77 位讀者各 40 篇，每個讀者與篇章只有一筆。40 篇各有兩種 `Truth` 條件。 |
| [目標句時間 CSV](https://files.us.osf.io/v1/resources/a6xqk/providers/osfstorage/62bb62fd9e9e6601028a165e) | OSF v1，321,916 bytes。欄位包括 `Subject, Block, Truth, Answer, Reading_Time`，有 18,108 列和重複的讀者與區塊鍵，缺少可用來驗證逐句次序的欄位。 |
| [後續句時間 CSV](https://files.us.osf.io/v1/resources/a6xqk/providers/osfstorage/62bb62ea9f91f0011d88542d) | OSF v1，106,104 bytes。5,808 列剛好是 2,904 個讀者×區塊鍵各出現兩次，分別帶 `Answer=1` 和 `Answer=2`。依作者的測量說明，這一欄無法解讀為讀者在後續句的實際按鍵。 |
| [目標句正確性 R 腳本](https://files.us.osf.io/v1/resources/a6xqk/providers/osfstorage/62bb62d49f91f0011e885588) | OSF v1，1,303 bytes。讀入 CSV 後把正誤、材料類別和條件轉成因子再擬合。腳本沒有列出論文排除的讀者，也沒有解釋兩筆 `Accuracy=2`。 |

就 CSV 中的二元正確性值重算，`(Material_Set, Truth)` 四格的答對數與分母依次為 `(1,1)` 717／769、`(1,2)` 422／770、`(2,1)` 709／770、`(2,2)` 411／769。另兩列編為 `Accuracy=2`，分別位於 `(1,1)` 和 `(2,2)`。論文在排除四位從未回答「不一致」及兩位反應過快的讀者後，報告 **72 位**納入分析。目前可重現的是公開表的 77 人分母。論文的採樣和清理分母應由其報告另行引用。 [原論文 Experiment 2，Results](https://doi.org/10.3758/s13421-022-01351-w) · [正確性原表](https://files.us.osf.io/v1/resources/a6xqk/providers/osfstorage/62bb62db9e9e6600fe8a13a3)

## 效應與材料的適用範圍

論文 Table 3 的正確率在概念故事的一致和不一致條件分別為 **93.4% 和 54.8%**，在人物特質故事為 **92.1% 和 53.6%**。這相當於兩組故事在矛盾目標句分別有 54.8% 和 53.6% 回報矛盾，而在一致目標句誤報矛盾的比率為 6.6% 和 7.9%。這些數字描述目標句當下的回報，無法排除同一讀者曾在前句回報。 [原論文 Experiment 2，Table 3](https://doi.org/10.3758/s13421-022-01351-w)

實驗二沿用實驗一的 40 篇故事，包括源自 Singer（2006、2017）的 20 篇概念故事，以及源自 O’Brien 等人（1998）的 20 篇人物特質故事。實驗三也由這套材料改編，把原目標句換成查問前文資訊的問題。已查核 OSF 的三個實驗目錄，包含資料表和分析腳本。論文 Table 1 提供兩篇示例，附錄 A 提供問卷。完整的 40 篇呈現文字及連到資料表 `Passage` 編碼的對照表尚未取得。 [原論文 Experiment 1–3，Materials、Table 1 及附錄 A](https://doi.org/10.3758/s13421-022-01351-w) · [作者 OSF 存檔](https://osf.io/a6xqk/)

Frick 與 Schüler（2026）以 Singer 的作業和結果作為實驗設計及統計力估算參照，並明列自己的材料為改編自 Cook（2014）的 12 篇及 Frick 等人（2025）的 8 篇。兩篇論文所列材料來源有別。逐題去重仍需 Singer 的完整材料。現有出版效應可作為跨研究的量級參照。若用它調整模型的效應幅度，同一統計量即成為校準依據。 [Frick 與 Schüler，Experiment 1 Design and Materials](https://doi.org/10.3758/s13421-026-01900-7) · [Singer 等人的實驗](https://doi.org/10.3758/s13421-022-01351-w)

較接近的 Schüler 與 Frick（2024）則以德語文字配圖，要求受試者依提示單詞按鍵。文字與圖片是否矛盾是操弄，反應時間是主要觀測之一。這提供跨模態干擾的比較，端點和 Singer 的明示矛盾判斷不同。 [原論文及公開材料入口](https://doi.org/10.1016/j.learninstruc.2024.101956)
