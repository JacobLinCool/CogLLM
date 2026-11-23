# 一次前向之後的同疑問狀態計算

新增可微分的疑問狀態計算。輸入是同一次因果前向所取特徵經預測頭算出的各 token 轉移分數，以及每個疑問何時可見的遮罩。每題同時保留尚未出現、未解、暫擱和已解四種狀態的機率。讀出包含每個 token 的狀態機率、各轉移事件的機率，以及第一次解開的時間分布。解開後再重開仍可透過已解到未解的轉移呈現。計算使用 log 空間的平衡矩陣乘積。輸出在讀出層完成，沒有把估計狀態回餵至 Qwen。 [程式](question_states.py) · [數學定義](../mathematical-formulation.md#5-同一疑問何時維持解開或重新出現)

同一題的狀態可由確定觀測、只排除某些狀態的觀測或序位回報約束。缺失時遮罩掉的是觀測，疑問本身仍可在已讀位置轉移。第一個解開的區間用首次進入已解狀態的機率，直到最後仍未觀察到則保留右設限。即使一題後來重開，第一次進入已解的分布也維持其原意。 [TED-Q 觀測缺失的實例](tedq-full-answer-bounds.md)

## 與既有封存分數核對

將原三組首次解答 hazard 視為新狀態計算的一個特例，只允許未解到已解，已解後保持已解。重播三篇故事的 **196 個疑問、588 條封存預測路徑**。其中 183 個疑問有觀測區間，含 6 個跨行區間，另 13 個右設限。用首次解答分布評分、用觀測狀態評分、以及直接依機率相乘的獨立算式結果一致，最大誤差低於 **4.5 × 10⁻¹⁵ nats**。各篇的平均誤差和舊分數一致，兩次 CPU 重播逐位元相同。 [原始實驗](../semantic-joint-first-run.md) · [重播程式](question_states_replay.py) · [封存數值](../results/question-states-interval-replay.json)

在小型測例中，直接枚舉所有狀態路徑，核對遮罩、首度解開、並行追蹤、重開、梯度及長序列的數值穩定性。把後文替換或截掉時，共有前綴的狀態和第一次解開機率逐值一致。另用 8 條、每條 8,192 token 的合成軌跡檢查前綴。FP32 的狀態機率正規化誤差不超過 7.1 × 10⁻⁶，FP64 不超過 1.3 × 10⁻¹⁴。 [獨立枚舉驗證](verify_question_states.py)

## 暫擱與重開的觀測範圍

Onea／PITQ 重計有 16 個原文行同時開啟一題並首次關閉另一題，支持一個位置可以有多個不同疑問的不同變化。該量表的 `?offen` 是標註者不確定該處是否已有解答，同題結束後依指引清除後續標記，因此本次資料只校驗既有首次閉合區間。重開是狀態模型已支援的轉移，仍缺同一讀者對同一題的實測序列。新計算已完成程式及舊端點重播，模型權重與 Gradio 的實驗性顯示維持原驗收狀態。 [原始工作簿核對](onea-reopening-source.md)

從專案根目錄用有 PyTorch 的 Python 執行以下指令。重播只讀原公開結果中的逐行機率及觀測區間，不需載入 Qwen。

```sh
python3 -m research.semantic_tracking.verify_question_states
python3 -m research.semantic_tracking.question_states_replay \
  --source research/results/semantic-joint-first-run \
  --output /tmp/question-states-interval-replay.json
```
