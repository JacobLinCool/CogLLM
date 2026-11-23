"""Offer an earlier phrase only after the writer selects a pronoun that refers back."""
from research.token_readout.app import source_html

STYLE = """
.writing-review .ref-review { color:#23352f!important; border-top:1px solid #cbd7cf; padding-top:.8rem; }
.writing-review .ref-review h3 { font-size:1.08rem; color:#23352f!important; }
.writing-review .ref-review details { margin:.35rem 0; }
.writing-review .ref-review summary { cursor:pointer; padding:.25rem 0; color:#273d36!important; }
.writing-review .ref-review p { color:#273d36!important; }
.writing-review .ref-answer { border-left:3px solid #356591; padding-left:.65rem; }
"""

def result(text, record):
    if record.get('scope')!='given_prior_reference':
        raise ValueError('This reference tool requires a writer-selected backward reference.')
    candidates=list(map(tuple,record['pronouns']))
    if candidates!=sorted(set(candidates)):
        raise ValueError('Reference choices must follow their original positions.')
    by_pronoun={}
    for link in record['links']:
        q=link['pronoun_start'],link['pronoun_end'];a,b=link['phrase_start'],link['phrase_end']
        if (link['scope']!='given_prior_reference' or q not in candidates or q in by_pronoun
            or not 0<=a<=link['antecedent_start']<link['antecedent_end']<=b<=q[0]):
            raise ValueError('Reference and earlier phrase must belong to the same article prefix.')
        by_pronoun[q]=(a,b)
    chunks=[];spans=[];rules=[]
    for i,(a,b) in enumerate(candidates):
        if type(a) is not int or type(b) is not int or not 0<=a<b<=len(text):
            raise ValueError('Invalid pronoun in the original draft.')
        start=max(0,a-32);end=min(len(text),b+23)
        display=('…' if start else '')+source_html(text[start:a])+'<strong>'+source_html(text[a:b])+'</strong>'+source_html(text[b:end])+('…' if end<len(text) else '')
        spans.append((a,b,'ref-pronoun-'+str(i)))
        rules.append(f'.writing-review:has(details.ref-{i}[open]) .ref-pronoun-{i} {{outline:2px solid #5c3d8b; outline-offset:.06em;}}')
        if (a,b) in by_pronoun:
            x,y=by_pronoun[a,b];spans.append((x,y,'ref-earlier-'+str(i)))
            rules.append(f'.writing-review:has(details.ref-{i}[open]) .ref-earlier-{i} {{box-shadow:inset 0 -.22em #94bddf;}}')
            answer=f'Likely earlier phrase · <strong>“{source_html(text[x:y])}”</strong>'
        else:answer='The model has too little evidence to choose one earlier phrase.'
        chunks.append(f'<details class="ref-{i}" name="cogllm-selected-reference"><summary>{display}</summary><p class="ref-answer">{answer}</p></details>')
    explain=('Choose a pronoun you intend to point back. Opening it marks that pronoun and the suggested earlier phrase in your draft.'
             if chunks else 'This draft has no supported English pronoun with earlier text to inspect.')
    return spans,('<style>'+''.join(rules)+'</style>' if rules else '')+('<section class="ref-review" aria-label="Inspect a reference"><h3>Which earlier phrase does this pronoun refer to?</h3>'
             f'<p>{explain}</p>'+''.join(chunks)+'</section>')
