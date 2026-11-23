"""Present the whole-summary human judgment in the units used by its annotators."""
from html import escape
from math import isfinite

SCOPE = "english_generated_summary_comprehensibility"
EXAMPLE = ("The city will replace the footbridge after inspectors found cracks in its supports. "
           "A shuttle will carry pedestrians across the river during the repairs.")


def panel(record):
    if record["scope"]!=SCOPE or not isinstance(record["available"],bool):
        raise ValueError("Unknown summary assessment.")
    threshold=record["threshold"]
    if not isinstance(threshold,(int,float)) or not isfinite(threshold) or not 0<threshold<1:
        raise ValueError("Invalid review threshold.")
    heading='<h3>Can a reader understand this summary?</h3>'
    if not record["available"]:
        if record["probability_difficult"] is not None or record["review"] is not None or not record["reason"]:
            raise ValueError("An unavailable reading must have an explanation.")
        body='<p>'+escape(record["reason"])+'</p>'
    else:
        p=record["probability_difficult"]
        if (not isinstance(p,(int,float)) or not isfinite(p) or not 0<=p<=1 or
                not isinstance(record["review"],bool) or record["review"]!=(p>=threshold)):
            raise ValueError("Inconsistent summary probability and review decision.")
        call="Worth a second read." if record["review"] else "Below the study’s review threshold."
        body=(f'<p><strong>{p:.0%}</strong> estimated chance that a study rater would say '
              f'they could not read and understand the whole summary. {call}</p>'
              '<p class="reading-caption">The study used short English summaries of news and how-to articles. '
              'The percentage concerns the whole draft. Sentence colors above use separate reader ratings. '
              '<a href="https://aclanthology.org/2023.emnlp-main.584/" target="_blank" '
              'rel="noopener noreferrer">Original reading study</a>.</p>')
    return '<section class="summary-understanding">'+heading+body+'</section>'
