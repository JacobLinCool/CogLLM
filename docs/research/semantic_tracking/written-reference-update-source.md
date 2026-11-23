# Incremental update of the human reference during stepwise text revelation

Degen, Kursat, and Leigh's Experiment 1 (2021) records four reference choices from the same reader within the same trial. The display starts at the beginning of the instruction and reveals the next segment after each keypress. The researchers specify the target referent, and participants select an on-screen candidate at each stop. This directly measures whether readers switch to the correct object after new text appears and when readers who previously chose correctly switch away. [Original study, Experiment 1](https://escholarship.org/content/qt6182t9jb/qt6182t9jb_noSplash_30a7058481c8315ffd8a880d943f55e5.pdf#page=3) · [Trial presentation and recording code](https://github.com/thegricean/eyetracking_replications/blob/b99241562dc6a1eea603b67e7da5a33fbc25bed0/experiments/SunBreheny/1_incremental/main/list1/public/js/index.js#L184)

The authors' `trials_merged.csv` contains 6,480 trials. The 4,320 main trials marked `ExpFiller == "Exp"` each contain four choices. `instruction3` preserves the full instruction. The presentation code reconstructs the four stops and punctuation from the prefixes of three, six, nine, and ten words. `response` records the four screen regions selected in order. `target1/target2` give the two selectable regions for the correct referents, and `location1` through `location10` describe the candidate display. The CSV's SHA-256 is `4f0291cd31e10f6f2bdcc9358d2cb33be7ffc62495666bcb8d6573ed425dea2d`. The release contains 119 distinct anonymous reader values, one with twice the number of trials. The following table uses the original CSV rows and includes trials regardless of sentence-final accuracy. [Pinned trial file](https://github.com/thegricean/eyetracking_replications/blob/b99241562dc6a1eea603b67e7da5a33fbc25bed0/analysis/SunBreheny/1_incremental/main/data/trials_merged.csv) · [Authors' classification code](https://github.com/thegricean/eyetracking_replications/blob/b99241562dc6a1eea603b67e7da5a33fbc25bed0/analysis/SunBreheny/1_incremental/main/rscripts/analysis.R)

| The readout position | Selected the target, accounting for all trials. | The proportion of trials where the target was selected from the previous reading stop, relative to the proportion of trials where the non-target was selected from the previous reading stop. | Leaving the target from the previous reading stop, accounting for the proportion of targets selected from the previous reading stop. |
| --- | ---: | ---: | ---: |
| Instruction prefix | 1,053／4,320（24.4%） | — | — |
| Gendered terms | 1,921／4,320（44.5%） | 989／3,267（30.3%） | 121／1,053（11.5%） |
| Quantifiers and proper noun phrases | 2,918／4,320（67.5%） | 1,240／2,399（51.7%） | 243／1,921（12.6%） |
| Noun | 3,812／4,320（88.2%） | 956／1,402（68.2%） | 62／2,918（2.1%） |

Of the 71 distinct full instructions, 18 varied across different scenes and mapped to different correct positions. To test whether the model replicates human choice updates, inputs must include the contemporary candidate scenes and be scored on four original text reading stops. Candidate scenes can be reconstructed from the stimulus program, with the complete trial serving as the comparison unit. This endpoint encompasses choice changes within the specified reference task. Questions raised in natural reading, earlier text references in the body, and the timing of reader confirmation still require their respective report data. The GitHub project does not list data or stimulus licenses, while the OSF preregistration indicates `No license`. Currently, only publicly available research results and verifiable data structures exist, leaving model reuse rights unconfirmed. [Fixed Stimulus](https://github.com/thegricean/eyetracking_replications/blob/b99241562dc6a1eea603b67e7da5a33fbc25bed0/experiments/SunBreheny/1_incremental/main/list1/public/js/stimuli.js) · [OSF Preregistration](https://api.osf.io/v2/registrations/vfgc8/) · [OSF License Field](https://api.osf.io/v2/licenses/563c1cf88c5e4a3877f9e965/)

## Two public sources of written narratives

- The fictitious theft scenario by Albertazzi et al. (2026) prompted each reader to re-evaluate the same suspect after every clue. Across 379 participants, each completed 11 full rounds, with inter-rater agreement verified via column-wise cross-checking in Stata and Excel. Equal weighting was applied per participant. The story group exhibited a 76.33% shift in ratings after reading non-diagnostic clues, compared to 61.66% in the no-new-information notification condition. In the story group, 17 readers returned to a score of 0.50 at least twice from other scores. The oTree code and per-round text identifiers from the main experiment were not recoverable, and the published `clue_length` were insufficient to identify the text presented during that round. Cross-checked against the original paper, registration, and OSF trial materials, this source was used for within-subject behavioral comparisons. [Paper and Appendix A](https://arxiv.org/abs/2607.04753v4) · [Fixed version cleaned data](https://github.com/paolopin/narrative-belief-updating/blob/9c9a68c1494c65c123c4c48e72bfda66a19c412f/Stata/data_cleaned.dta) · [Per-file and aggregated re-computation](../../../research/results/albertazzi-narrative-source-audit.json)
- **Beukers et al. (2024)** instructed readers to read six short stories sentence-by-sentence and predict the next event at two distinct positions. The original Methods section lists the sentences for each state. for instance, the authors' `data/human/inserted_late_rep_thresh90.csv` includes readers, stories, positions, states, candidate transitions, and actual selections. Two detection tasks, each targeting a different next event, allow for an examination of how prior story cues and learned regularities alter subsequent predictions. The sealed snapshot version on Zenodo is licensed under CC BY 4.0. [Paper and Materials and Methods](https://doi.org/10.1038/s44271-024-00079-4) · [Sequential Data](https://github.com/PrincetonCompMemLab/blocked_training_facilitates_learning/blob/eb0edbe51042ffaf065bd90839f4f999de5f3ffb/data/human/inserted_late_rep_thresh90.csv) · [License and Provenance](https://zenodo.org/api/records/10695055)

## Beukers' reader background experiment

[This study found] that **controlled learning trajectories can be validated for their impact on written story prediction**. The `sealed snapshot` version `mint` corresponds to submission `eb0edbe51042ffaf065bd90839f4f999de5f3ffb`. The original sample comprised 33 readers in the block group and 39 in the interleaved group, while the independent re-run sample included 26 and 24 readers, respectively. all were filtered by author attention. All visible coffee shop records for stories #0–159 conform to the group allocation. In the block group, shops were cycled every 40 stories, whereas in the interleaved group, shops alternated every story, with both starting orders implemented. This allows reconstruction of the assigned 80/80 exposure schedule. Stories replaced by attention items in both questions may be missing. Each person and each shop still has 106–141 available early prediction responses, usable for estimating pre-test proficiency. The internal story structure adopted remains a potential state. [Paper Method](https://gershmanlab.com/pubs/Beukers24.pdf#page=11) · [CC BY 4.0 sealed snapshot data](https://zenodo.org/api/records/10695055) · [Fixed human trial file](https://github.com/PrincetonCompMemLab/blocked_training_facilitates_learning/tree/eb0edbe51042ffaf065bd90839f4f999de5f3ffb/data/human)

We recovered the same fictional person's name and coffee shop from `true_rfc` and `false_rfc`, and reconstructed the prefix from visible nodes and the original sentence templates. For the second probe, we retained only stories that could be linked to the first probe. We used stable fields for the source nodes, shops, and choices, and used the CSV's `true_tonode` to identify the correct transition. The blocked and interleaved groups shared 692 prefixes in the original sample and 475 in the replication sample matching on name, shop, visible nodes, and the two candidate events. Also matching left–right choice order left **585 matched sets** in the original sample, with 784 blocked and 825 interleaved trials, and **347 matched sets** in the replication sample, with 449 and 426 trials, respectively. The groups cover eight shared visible prefix structures, allowing evaluation across the original and replication participant samples. Generalization to new story structures remains untested. [Node mapping](https://github.com/PrincetonCompMemLab/blocked_training_facilitates_learning/blob/eb0edbe51042ffaf065bd90839f4f999de5f3ffb/humanUtils.py#L10) · [Story templates in the paper](https://gershmanlab.com/pubs/Beukers24.pdf#page=12)

The pre-specified observation is the probability of selecting a candidate event. The first probe fixed at reading `NODE11` compares the probability difference of selecting `NODE21` when shifting the earlier coffee shop cue from Jungle to Deep Ocean. Using reader equal weighting, the original sample shows block group and interleaved group increases of approximately 71 and 14 percentage points, respectively, while the redo sample shows increases of approximately 79 and 15 percentage points. The model is developed using only the original sample and then evaluated on the independent redo sample, where name, prefix, and option order are held constant across slices. The comparison targets the "prefix and options," "additional learning group additive effect," and "learning group by text cue interaction." Lower negative log-likelihood of candidate probability is better. If the interaction readout reproduces human behavior, it should simultaneously reduce error in the redo data and reproduce the larger cue effect in the block group. If personal early performance is included, only the first 160 responses are used. This measures the selection of the next event and the learning history, with two probes asking about different events.

<details>
<summary>Recalculate settings and Python 3 standard library code</summary>

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

### Independent materials with different insertion orders

The same study includes three insertion positions within the first 160 items: beginning, middle, and end. The original samples comprised 21, 20, and 24 readers, respectively, while the repeated experiments involved 22, 24, and 12 readers, respectively. All six release files can reconstruct the readout prefix. two candidate files correspond to left and right positions. Published repeated experiments by the authors indicate that the early-insertion group achieved higher accuracy than the middle and end groups. New data anonymized IDs are encoded per file, preventing cross-file individual verification. the methodology treats repeated experiments as independently recruited. [Original Study Fig. 10](https://gershmanlab.com/pubs/Beukers24.pdf#page=9) · [Fixed Six Human Trial Files](https://github.com/PrincetonCompMemLab/blocked_training_facilitates_learning/tree/eb0edbe51042ffaf065bd90839f4f999de5f3ffb/data/human)

Using only the original text of the previously extracted representations, with names, visible prefix, and left/right options consistent, the original experiment yielded 287 early-mid layouts and 303 early-late layouts. Repeated experiments produced 316 and 173 layouts, respectively. All pairings retained all readers involved in the respective groups. This verification did not utilize responses from the last 40 items of these six batches. The two background encodings available from the sealed snapshot readout can be used to verify whether "performing block practice first" is sufficient for transfer. this encoding groups mid and late into the same value, preventing testing of divergent responses between them. [Scope of background readout](beukers-background-results.md#post-hoc-comparison-of-readout-selection-and-cue-identification)

[Original research record](../../../research/semantic_tracking/written-reference-update-source.md)
