"""Audit all original English-held SEAHORSE drafts against materials used in earlier study fits."""
from research.reading_validation.emtec_material import exposures
from research.reading_validation.transition_seahorse_data import DEST, source_rows
from research.reading_validation.transition_seahorse_features import save


def main():
    rows=list(source_rows("test"))
    texts=sorted({r["summary"] for r in rows})
    matched, details=exposures(texts)
    known={texts[i] for i in matched}
    groups=sorted({r["gem_id"] for r in rows if r["summary"] in known})
    save(DEST/"historical-exposure.json", {"detectors":"Prior registered exact, 12-word and near-duplicate checks",
         "checks":details, "source_articles_with_matches":groups, "matching_unique_test_texts":len(known),
         "total_unique":len(texts),"grades_used":False})
    print("Known in earlier SE study",len(known),len(groups),flush=True)


if __name__=="__main__":
    main()
