"""Optional English reader-reference review from the writer's original article forward."""

from contextlib import contextmanager
from dataclasses import dataclass

from research.reading_validation.antecedent_action_inference import ApplicableAntecedentInference
from research.reading_validation.antecedent_inference import AntecedentInference,BUNDLE
from research.reading_validation.antecedent_phrase_inference import PHRASE_MODEL,PHRASE_MODEL_SHA,PhraseInference,expand_links


@dataclass(frozen=True)
class ReferenceInference:
    location: object
    phrases: PhraseInference
    scope: str

    @classmethod
    def conditional(cls,directory=BUNDLE):
        """Investigate an earlier link when the reader is understood to refer backwards."""
        return cls(AntecedentInference.load(directory),PhraseInference.load(PHRASE_MODEL,PHRASE_MODEL_SHA),"given_prior_reference")

    @classmethod
    def applicable(cls,signature,directory=BUNDLE):
        """Use the all-action policy certified for ordinary English when its release rule qualifies."""
        return cls(ApplicableAntecedentInference.load(signature,directory),
                   PhraseInference.load(PHRASE_MODEL,PHRASE_MODEL_SHA),"all_reader_actions")

    @contextmanager
    def capture(self,text,runtime):
        with self.location.capture(text,runtime) as observation:
            pointer=observation.pointer if self.scope=="all_reader_actions" else observation
            yield ReferencePass(text,observation,pointer,self.phrases,self.scope)


@dataclass
class ReferencePass:
    text:str
    observed:object
    pointer:object
    phrases:PhraseInference
    scope:str

    def links(self,encoded):
        selected=self.observed.links(encoded)
        found=expand_links(self.text,self.pointer,selected,self.phrases)
        for r in found:
            r["scope"]=self.scope
        return found
