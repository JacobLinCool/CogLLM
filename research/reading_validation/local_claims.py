"""The fixed, directional inventory for the local-difficulty multiplicity audit."""

from dataclasses import dataclass
from pathlib import Path

RESULTS = Path("research/results")


@dataclass(frozen=True)
class Claim:
    study: str
    label: str
    artifact: str
    path: tuple[str, ...]
    direction: int = 1

    @property
    def file(self):
        return RESULTS / ("reading-validation-" + self.artifact + ".json")


def claim(study, label, artifact, path, direction=1):
    return Claim(study, label, artifact, tuple(path.split("|")), direction)


def inventory():
    c = []
    place = "local/localization/scores"
    c.append(
        claim(
            "PLABA",
            "Named phrases, top five vs form",
            place,
            "paired|sentence|hit5|local minus form",
        )
    )
    c.append(
        claim(
            "PLABA",
            "Named phrases, top one vs form",
            place,
            "paired|sentence|hit1|local minus form",
        )
    )
    c.append(
        claim(
            "PLABA",
            "Whole-article phrase location vs form",
            "local/article-localization/scores",
            "metrics|hit5|local_joint minus form_joint",
        )
    )
    for endpoint in ("ease", "understanding"):
        c.append(
            claim(
                "PLABA",
                f"Sentence {endpoint} loss vs earlier head",
                "local/ratings/scores",
                f"metrics|{endpoint}|nll|full_state_local minus historical_frozen",
                -1,
            )
        )
    c.append(
        claim(
            "Kerwer",
            "Predicted heading effect above zero",
            "local/headings/scores",
            "estimates|local|model_effect",
        )
    )
    for control in ("form", "external_lexical"):
        c.append(
            claim(
                "Guo",
                f"Postreading familiarity vs {control}",
                "local/jargon/scores",
                f"after|estimates|local minus {control}",
            )
        )
    c.append(
        claim(
            "Guo",
            "Adding text to a reader's prior, ordinal loss",
            "local/jargon/crossfit-scores",
            "after|metrics|nll_nats|text_with_prior minus prior",
            -1,
        )
    )
    c.append(
        claim(
            "Guo",
            "Adding a reader's prior for requested definitions",
            "local/help-choice/scores",
            "heldout_comparisons|definition|all_priors|concordance|lexical_prior minus lexical",
        )
    )
    c.append(
        claim(
            "Guo",
            "Article evidence for definition versus example at equal prior",
            "local/help-choice/scores",
            "heldout_comparisons|definition_over_example|same_prior|concordance|article_prior minus lexical_prior",
        )
    )
    for cohort in ("l2", "dhh"):
        c.append(
            claim(
                "Isolated words " + cohort,
                "Within-reader Spearman gain vs surface",
                "local/word-readers/scores",
                cohort + "|strict_novel|results|qwen minus surface",
            )
        )
    for control in ("length", "edit"):
        c.append(
            claim(
                "SimpDA",
                f"Simplicity order vs {control}",
                "revision/scores",
                "group_mean_preferences|simplicity|crossed_source_reader_interval|full minus "
                + control,
            )
        )
    c.append(
        claim(
            "SimpDA",
            "Both winners among opposed preferences vs trained edit scores",
            "revision/contrasts",
            "opposed_preferences|crossed_95|adaptive minus edit",
        )
    )
    for method in (
        "retained_length",
        "edit_meaning",
        "source_word_recall",
        "simplicity",
    ):
        c.append(
            claim(
                "OneStopQA",
                f"Observed correctness, meaning vs {method}",
                "comprehension/scores",
                "primary|correctness|paired|meaning minus " + method,
            )
        )
    c.append(
        claim(
            "Leroy",
            "Sentence-difficulty squared error vs form",
            "noun/scores",
            "primary_family_stratified_mse_gain",
        )
    )
    c.append(
        claim(
            "ReLay",
            "Within-reader simplicity vs four readability formulas",
            "relay-summary/scores",
            "datasets|primary|frozen|simplicity|differences|plaba_ease_minus_authors_formula",
        )
    )
    c.append(
        claim(
            "ReLay",
            "Simplicity, adaptive transferred coefficients vs target only",
            "relay-summary/collective-scores",
            "endpoints|simplicity|paired_improvement|adaptive_over_independent",
        )
    )
    c.append(
        claim(
            "Paetzold unknown",
            "Word-report rank vs trained lexical reference",
            "nns/scores",
            "reported|rank|paired_difference|adaptive_minus_lexical",
        )
    )
    c.append(
        claim(
            "Paetzold unknown",
            "Word-report log loss, adding a reader's CEFR",
            "nns/scores",
            "reported|logloss|paired_difference|text.adaptive_minus_cefr.adaptive",
        )
    )
    for study, endpoint in (
        ("Paetzold unknown", "unknown_word"),
        ("Paetzold preference", "revision_preference"),
    ):
        c.append(
            claim(
                study,
                "Joint vs matched independent likelihood",
                "nns-rewrite/scores",
                endpoint
                + "|logloss|oriented_gains|contextual.adaptive_gain_vs_contextual.independent",
            )
        )
    for endpoint in ("passage_rps", "word_reread_logloss"):
        c.append(
            claim(
                "EMTeC",
                endpoint + ", added unresolved stock vs strong common control",
                "emtec/scores",
                endpoint + "|gain_over_strong",
            )
        )
    for method in (
        "context_invariant",
        "lexical_plus_surprisal",
        "paper_adaptive_ngram",
    ):
        c.append(
            claim(
                "SemEval",
                "Context reversal vs " + method,
                "semlex/scores",
                "all_shared_candidate_pairs|relative_confidence|contrasts|mix_minus_"
                + method,
            )
        )
    if len(c) != 32:
        raise ValueError(
            "The published local multiplicity inventory contains 32 comparisons"
        )
    return c
