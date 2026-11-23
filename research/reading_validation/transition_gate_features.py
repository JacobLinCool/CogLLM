"""Features available when the current English pronoun has just been read."""

import numpy as np

from research.reading_validation.transition_pointer_features import FORMS, PARTS


FORMS_ORDER = tuple(sorted(FORMS))
PREVIOUS = ("a","an","the","and","but","or","as","because","when","while","if","that","which",
            "so","for","to","in","on","with","of","said","is","was","then")
NAMES = tuple("current_"+w for w in FORMS_ORDER) + tuple("previous_"+w for w in PREVIOUS) + tuple(
    "previous_pos_"+tag for tag in PARTS) + ("position","words_since_sentence_start","sentence_index",
    "first_in_sentence","uppercase","previous_same_form","previous_same_class","prior_nouns","prior_proper_nouns",
    "prior_pronouns","nearest_noun","nearest_proper_noun","nearest_pronoun","preceded_by_quote","preceded_by_comma")


def features(row, q, tags):
    text = row["text"]
    words = [text[a:b] for a,b in row["words"][:q+1]]
    forms = [s.casefold() for s in words]
    current = forms[-1]
    if current not in FORMS or len(tags) < q:
        raise ValueError("An antecedent existence forecast needs prefix-only pronoun and POS data")
    parts = np.asarray(tags[:q])
    recent = parts[-1] if parts[-1] in PARTS else "other"
    nouns=np.array([s.startswith("NN") for s in parts]);proper=np.isin(parts,["NNP","NNPS"])
    prons=np.array([s.startswith("PRP") for s in parts]);sentence=row["sentences"][q]
    sentence_length=int(np.sum(np.asarray(row["sentences"][:q])==sentence))
    def distance(mask):
        have=np.flatnonzero(mask)
        return np.log1p(q-have[-1]) if len(have) else 0.0
    gap=text[row["words"][q-1][1]:row["words"][q][0]]
    values=[current==w for w in FORMS_ORDER]+[forms[-2]==w for w in PREVIOUS]+[recent==p for p in PARTS]+[
        np.log1p(q),np.log1p(sentence_length),np.log1p(sentence),sentence_length==0,words[-1][0].isupper(),
        np.log1p(forms[:-1].count(current)),np.log1p(sum(FORMS.get(v)==FORMS[current] for v in forms[:-1])),
        *(np.log1p(m.sum()) for m in (nouns,proper,prons)),*(distance(m) for m in (nouns,proper,prons)),
        any(c in gap for c in ('"', '“', '”', '‘', '’')),"," in gap]
    z=np.asarray(values,dtype=np.float32)
    if z.shape!=(len(NAMES),) or not np.isfinite(z).all():
        raise ValueError("Unexpected discourse-action gate geometry")
    return z
