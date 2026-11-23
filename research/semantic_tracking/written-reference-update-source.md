# 逐步揭示文字時的真人指涉更新

Degen、Kursat 與 Leigh（2021）實驗一提供同一讀者在同一試次的四次指涉選擇。畫面從指令開頭開始，每按一次才揭示下一段文字。研究者指定要找的指涉對象，參與者在每個停點從畫面上的候選處點選。這讓我們直接量測新文字出現後，讀者是否改選正確對象，以及原先選對的人何時又改選。 [原研究實驗一](https://escholarship.org/content/qt6182t9jb/qt6182t9jb_noSplash_30a7058481c8315ffd8a880d943f55e5.pdf#page=3) · [試次呈現與紀錄程式](https://github.com/thegricean/eyetracking_replications/blob/b99241562dc6a1eea603b67e7da5a33fbc25bed0/experiments/SunBreheny/1_incremental/main/list1/public/js/index.js#L184)

作者的 `trials_merged.csv` 有 6,480 筆試次，其中 4,320 筆 `ExpFiller == "Exp"` 的主要試次各有四次選擇。`instruction3` 保留完整指令，呈現程式依前三、六、九、十詞還原四個停點及顯示標點。`response` 是四個依序選取的畫面區域，`target1/target2` 是正確對象的兩個點選區域，`location1` 至 `location10` 記錄候選畫面。CSV 的 SHA-256 為 `4f0291cd31e10f6f2bdcc9358d2cb33be7ffc62495666bcb8d6573ed425dea2d`。發布檔的匿名讀者欄有 119 個不同的值，其中一個含兩倍的試次。下表以 CSV 原始列為單位，未依句末答對與否篩選。 [固定版試次檔](https://github.com/thegricean/eyetracking_replications/blob/b99241562dc6a1eea603b67e7da5a33fbc25bed0/analysis/SunBreheny/1_incremental/main/data/trials_merged.csv) · [作者的分類程式](https://github.com/thegricean/eyetracking_replications/blob/b99241562dc6a1eea603b67e7da5a33fbc25bed0/analysis/SunBreheny/1_incremental/main/rscripts/analysis.R)

| 已讀到的位置 | 選到目標，占全部試次 | 從前一停點改選目標，占前一停點選非目標者 | 從前一停點離開目標，占前一停點選目標者 |
| --- | ---: | ---: | ---: |
| 指令開頭 | 1,053／4,320（24.4%） | — | — |
| 性別片語 | 1,921／4,320（44.5%） | 989／3,267（30.3%） | 121／1,053（11.5%） |
| 量詞與人名片語 | 2,918／4,320（67.5%） | 1,240／2,399（51.7%） | 243／1,921（12.6%） |
| 名詞 | 3,812／4,320（88.2%） | 956／1,402（68.2%） | 62／2,918（2.1%） |

71 種不同的完整指令中，有 18 種在不同畫面對應到不同的正確位置。若要測試模型是否重現人的選擇更新，輸入需要包含當時的候選場景，並以四個原始文字停點評分。可以從刺激程式重建候選畫面，以完整試次作比較單位。這個端點涵蓋已指定指涉任務中的選擇改變。自然閱讀中的疑問發起、正文中的上文指向，以及讀者何時確認解答，仍需要各自的回報資料。GitHub 專案未列資料與刺激授權，OSF 預註冊標示 `No license`。目前具備公開研究結果及可核對的資料結構，模型重用權仍待核明。 [固定版刺激](https://github.com/thegricean/eyetracking_replications/blob/b99241562dc6a1eea603b67e7da5a33fbc25bed0/experiments/SunBreheny/1_incremental/main/list1/public/js/stimuli.js) · [OSF 預註冊](https://api.osf.io/v2/registrations/vfgc8/) · [OSF 授權欄](https://api.osf.io/v2/licenses/563c1cf88c5e4a3877f9e965/)

## 書面敘事的兩個公開來源

- **Albertazzi 等人（2026）**的虛構竊案讓同一位讀者在每條線索後重評同一名嫌疑人。379 位參與者各 11 個完整回合，在 Stata 和 Excel 逐欄核對一致。逐人等權計算，故事組在讀到無診斷力的線索後有 76.33% 的評分變動，只有無新資訊通知的組別為 61.66%。故事組中有 17 位讀者至少兩次從其他分數回到 0.50。主實驗的 oTree 呈現程式及逐回合文字識別碼未查得，發布的 `clue_length` 也不足以辨認當次呈現的文字。已核對原論文、註冊和 OSF 的試行材料，此來源用於同命題的行為比較。 [論文及附錄 A](https://arxiv.org/abs/2607.04753v4) · [固定版清理資料](https://github.com/paolopin/narrative-belief-updating/blob/9c9a68c1494c65c123c4c48e72bfda66a19c412f/Stata/data_cleaned.dta) · [逐檔及彙總重算](../results/albertazzi-narrative-source-audit.json)
- **Beukers 等人（2024）**讓讀者逐句讀六句短故事，並在兩個不同位置預測下一個事件。原文 Methods 給出各狀態的句子；例如作者的 `data/human/inserted_late_rep_thresh90.csv` 有讀者、故事、位置、狀態、候選轉移及實際選擇。兩次探測各針對不同的下一事件，可檢驗先前故事線索及已學規律如何改變後續預測。Zenodo 的封存版本列 CC BY 4.0。 [論文及材料方法](https://doi.org/10.1038/s44271-024-00079-4) · [逐次資料檔](https://github.com/PrincetonCompMemLab/blocked_training_facilitates_learning/blob/eb0edbe51042ffaf065bd90839f4f999de5f3ffb/data/human/inserted_late_rep_thresh90.csv) · [授權與發行紀錄](https://zenodo.org/api/records/10695055)

## Beukers 的讀者背景實驗

〔本研究發現〕可以驗證**受控學習歷程對書面故事預測的影響**。封存版本 `mint` 對應提交 `eb0edbe51042ffaf065bd90839f4f999de5f3ffb`。原始樣本的區塊組與交錯組各有 33、39 位讀者，獨立重做樣本各有 26、24 位，均採作者注意力篩選後的檔案。0–159 號故事的所有可見咖啡店紀錄都符合各組的編排。區塊組每 40 篇換店，交錯組每篇交替，兩種起始順序都有。依此可重建指派的 80／80 篇曝光安排。兩題皆被注意力題取代的故事可能缺列。每人、每家店仍有 106–141 筆可用的早期預測答覆，能用來估計測試前掌握程度。內在採用了哪種故事結構仍是潛在狀態。 [論文方法](https://gershmanlab.com/pubs/Beukers24.pdf#page=11) · [CC BY 4.0 封存資料](https://zenodo.org/api/records/10695055) · [固定版真人試次檔](https://github.com/PrincetonCompMemLab/blocked_training_facilitates_learning/tree/eb0edbe51042ffaf065bd90839f4f999de5f3ffb/data/human)

從 `true_rfc`、`false_rfc` 取得同一個虛構人名和咖啡店，依可見節點及原文句型重建前綴。第二個探測只保留可連到前一探測的故事。使用穩定的來源節點、咖啡店和選項欄位，正確轉移依 CSV 的 `true_tonode`。兩組在原始樣本共有 692 組相同人名、咖啡店、可見節點及兩個候選事件的前綴，重做樣本有 475 組。若連左右選項順序也固定，分別有 **585 組**，涵蓋區塊組 784 筆、交錯組 825 筆，以及 **347 組**，涵蓋 449 與 426 筆。兩組共同覆蓋八種可見前綴結構，適合按原始與重做人員分割。跨新故事結構的泛化仍未受測。 [節點對照](https://github.com/PrincetonCompMemLab/blocked_training_facilitates_learning/blob/eb0edbe51042ffaf065bd90839f4f999de5f3ffb/humanUtils.py#L10) · [論文的故事句型](https://gershmanlab.com/pubs/Beukers24.pdf#page=12)

預先指定的觀測量為選擇某個候選事件的機率。第一個探測固定讀到 `NODE11`，比較把較早的咖啡店線索從 Jungle 改為 Deep Ocean 時，選擇 `NODE21` 的機率差。以讀者等權估計，原始樣本的區塊組與交錯組分別增加約 71 與 14 個百分點，重做樣本分別增加約 79 與 15 個百分點。模型只用原始樣本開發，再於獨立重做樣本檢查名字、前綴和選項順序都相同的切片。比較「前綴與選項」、「再加學習組別的加法效應」及「學習組別與文字線索的交互作用」。候選機率的負對數概似越低越好。若交互讀出重現真人行為，應在重做資料中同時降低誤差，並重現區塊組較大的線索效應。若加入個人早期表現，僅使用前 160 篇的作答。這裡測的是指定下一事件的選擇與學習史，兩次探測分別詢問不同事件。

<details>
<summary>重算設定與 Python 3 標準函式庫程式</summary>

```python
import csv
import hashlib
import io
from collections import Counter, defaultdict
from statistics import mean
from urllib.request import urlopen

commit = "eb0edbe51042ffaf065bd90839f4f999de5f3ffb"
base = f"https://raw.githubusercontent.com/PrincetonCompMemLab/blocked_training_facilitates_learning/{commit}/data/human/"
checksums = {
    "blocked": "9e813cd0ddfbe5a3bbc68169ecb7f961f7509798a58009cf08806fd1be69a7db",
    "interleaved": "c7c57447f7f0f20eeb86aa5356fa3d9f34c5d242cfb45b6214c582422ac0d642",
    "blocked_rep": "b7b9f2853a4b984b5b650bf36960e0689b14bd67e83e2de8059f2b1c505e89d8",
    "interleaved_rep": "c9047a6c6ba11a15f91729e47abae874f2381e8e40747a164ae72467cc0eb510",
}

exact, ordered = {}, {}
for cohort, checksum in checksums.items():
    with urlopen(f"{base}{cohort}_thresh90.csv", timeout=15) as stream:
        data = stream.read(1_750_001)
    assert len(data) <= 1_750_000 and hashlib.sha256(data).hexdigest() == checksum
    stories = defaultdict(dict)
    for row in csv.DictReader(io.StringIO(data.decode("utf-8-sig"))):
        assert row["qtype"] == "transQ" and row["true_rfc"] == row["false_rfc"]
        name, cafe = row["true_rfc"].removeprefix("subject-").split("_location-")
        assert cafe in ("deep", "jungle")
        story, depth = int(row["story"]), int(row["depth"])
        sequence = stories[row["subjnum"], story]
        assert depth in (1, 2) and depth not in sequence
        assert int(row["correct_response"]) == (row["response_right"] == row["true_on_right"])
        sequence[depth] = row, name, cafe

    orientations, early_probes, choices = defaultdict(set), Counter(), defaultdict(list)
    exact[cohort], ordered[cohort] = Counter(), Counter()
    for (reader, story), sequence in stories.items():
        if len(sequence) == 2:
            first, first_name, first_cafe = sequence[1]
            second, second_name, second_cafe = sequence[2]
            assert (first_name, first_cafe, first["true_tonode"]) == (second_name, second_cafe, second["fromnode"])
        for depth, (row, name, cafe) in sequence.items():
            if story < 160:
                parity = (story // 40 if cohort.startswith("blocked") else story) % 2
                orientations[reader].add((cafe == "jungle") ^ parity)
                early_probes[reader, cafe] += 1
                continue
            if depth == 1 and row["fromnode"] == "NODE11":
                chosen = row["true_tonode"] if row["correct_response"] == "1" else row["false_tonode"]
                choices[reader, cafe].append(chosen == "NODE21")
            if depth == 2 and 1 not in sequence:
                continue
            nodes = tuple(sequence[step][0]["fromnode"] for step in range(1, depth + 1))
            options = tuple(sorted((row["true_tonode"], row["false_tonode"])))
            right = row["true_tonode"] if row["true_on_right"] == "True" else row["false_tonode"]
            key = name, cafe, nodes, options
            exact[cohort][key] += 1
            ordered[cohort][(*key, right)] += 1

    assert all(len(values) == 1 for values in orientations.values())
    by_cafe = {
        cafe: mean(mean(answers) for (_, site), answers in choices.items() if site == cafe)
        for cafe in ("deep", "jungle")
    }
    print(cohort, len(orientations), min(early_probes.values()), max(early_probes.values()),
          round(100 * (by_cafe["deep"] - by_cafe["jungle"]), 1))

for blocked, interleaved in (("blocked", "interleaved"), ("blocked_rep", "interleaved_rep")):
    for tables in (exact, ordered):
        a, b = tables[blocked], tables[interleaved]
        keys = a.keys() & b.keys()
        families = {(key[1], key[2]) for key in keys}
        print(blocked, interleaved, len(keys), len(families),
              sum(a[key] for key in keys), sum(b[key] for key in keys))
```

</details>

### 不同插入順序的獨立材料

同一論文另有三種在前 160 篇中插入區塊的位置，分別在開頭、中間和末尾。其原始樣本各有 21、20、24 位讀者，重複實驗各有 22、24、12 位。六個發布檔均可重建已讀前綴、兩個候選與左右位置。作者已發表的重複實驗顯示，早插入組的答對率高於中間與末尾組。新資料的匿名 ID 各檔各自編碼，跨檔無法核對個人；研究方法將重複實驗記為獨立招募。 [原研究 Fig. 10](https://gershmanlab.com/pubs/Beukers24.pdf#page=9) · [固定版六份真人試次檔](https://github.com/PrincetonCompMemLab/blocked_training_facilitates_learning/tree/eb0edbe51042ffaf065bd90839f4f999de5f3ffb/data/human)

只用先前已擷取表徵的原名文字，且要求人名、可見前綴與左右選項一致，原始實驗可得到早對中 287 種版面、早對晚 303 種。重複實驗分別有 316 種和 173 種。各配對都保留所涉組別的全部讀者。此查核未使用這六批最後 40 篇的作答。現有封存讀出的兩種背景編碼可用來驗證「一開始先做區塊練習」是否足以移轉，這個編碼會把中與晚放在同一值，無法檢驗兩者相異的反應。 [背景讀出的作用範圍](beukers-background-results.md#線索辨識與選擇讀出的事後對照)
