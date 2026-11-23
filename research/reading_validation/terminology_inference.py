"""Writer-facing terminology events and expected counts from a shared encoder pass."""

from pathlib import Path

from research.reading_validation.definition_inference import causal_page
from research.reading_validation.definition_stock import evidence,render
from research.reading_validation.stock_inference import ExpectedConceptLedger,render_distribution
from research.reading_validation.stock_probability import predict


class TerminologyInference:
    def __init__(self,model):
        self.model=model

    @classmethod
    def load(cls):
        root=Path(__file__).resolve().parents[1]/"results/reading-validation-definitions"
        return cls(ExpectedConceptLedger.load(root))

    def score(self,text,features,runtime):
        head=self.model.ledger
        if len(features.token_ids)>1024:
            raise ValueError("The scientific terminology analysis accepts up to 1,024 model tokens per passage.")
        page=causal_page(text,features,runtime,head.model_revision)
        posterior=predict(page,"semantic",self.model.fit,(head.term,head.link))
        continuous=render_distribution(page,posterior,features.offsets,self.model.scale)
        hard=render(page,evidence(page,head.term,head.link,head.arm),head.threshold,features.offsets)
        words={w["token_index"]:w for w in hard["words"]}
        links=[]
        for mention in hard["mentions"]:
            t=mention["closing_token"]
            if t is None:
                continue
            cue=words[t]
            if cue["closed_mention"]!=mention["mention_id"] or not mention["chars"][1]<=cue["chars"][0]:
                raise ArithmeticError("A predicted definition does not follow its original terminology")
            links.append(dict(source=mention["chars"],cue=cue["chars"],score=cue["explanatory_evidence"]))
        return dict(expected=[r["expected_pending"] for r in continuous["tokens"]],
                    starts=[r["span"][0] for r in continuous["tokens"]],
                    ends=[r["span"][1] for r in continuous["tokens"]],
                    links=sorted(links,key=lambda link:(link["cue"],link["source"])),
                    tokens=len(features.token_ids),threshold=head.threshold)
