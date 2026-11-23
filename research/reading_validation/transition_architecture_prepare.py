"""Prepare the exact common word universe across three historical frozen tokenizers."""
from hashlib import sha256
import json
from pathlib import Path

from tokenizers import Tokenizer

from research.reading_validation.transition_snac_fit import inputs
from research.reading_validation.transition_snac_prepare import ROOT
from research.reading_validation.transition_score import dump, sha

RESULTS = Path('research/results/reading-validation-transition-architecture')
PROTOCOL = Path('research/reading_validation/transition-architecture-protocol.md')
CANDIDATES = ('qwen', 'lfm3', 'gemma4')


def load_tokenizer(name, spec):
    path = Path('.cache/cogllm/backbone-transfer/source') / name / 'tokenizer.json'
    if not path.is_file() and name == 'qwen':
        path = (Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen3.5-2B'
                / 'snapshots' / spec['revision'] / 'tokenizer.json')
    if sha(path) != spec['sha256']['tokenizer.json']:
        raise ValueError(f'Tokenizer bytes changed for {name}')
    return Tokenizer.from_file(str(path))


def word_indices(text, pairs, tokenizer, encoded):
    ends = {end: index for index, (_, end) in enumerate(encoded.offsets)}
    rows = []
    cursor = 0
    for left, right in pairs:
        while cursor < len(encoded.offsets) and encoded.offsets[cursor][1] <= left:
            cursor += 1
        end = ends.get(right)
        if (end is None or cursor > end or encoded.offsets[cursor][0] > left
                or tokenizer.encode(text[:right], add_special_tokens=False).ids != encoded.ids[:end+1]):
            rows.append(None)
        else:
            rows.append([cursor, end])
    return rows


def main():
    docs, meta = inputs()
    specs = json.loads(Path('research/backbone_transfer/models.json').read_text())['models']
    tokenizers = {name: load_tokenizer(name, specs[name]) for name in CANDIDATES}
    output = ROOT / 'cross-snac-stimuli.jsonl'
    support = {}
    with output.open('w') as handle:
        for doc_id, row in docs.items():
            text, words = row['text'], row['words']
            encodings = {name: tok.encode(text, add_special_tokens=False) for name,tok in tokenizers.items()}
            sizes = {name: len(value.ids) for name,value in encodings.items()}
            detail = {'tokens': sizes, 'original_words': len(words)}
            support[doc_id] = detail
            if max(sizes.values()) > 2048:
                detail['reason'] = 'complete_document_token_limit'
                continue
            loc = {name: word_indices(text, [w[:2] for w in words], tok, encodings[name])
                   for name,tok in tokenizers.items()}
            keep = [i for i in range(len(words)) if all(loc[name][i] is not None for name in CANDIDATES)]
            if not keep:
                raise ValueError('Eligible narrative has no common words')
            for i in keep:
                if loc['qwen'][i] != words[i][2:4]:
                    raise ValueError('Qwen range differs from the frozen earlier source')
            detail['accepted_words'] = len(keep)
            detail['uncertified'] = {name: sum(p is None for p in arr) for name, arr in loc.items()}
            print(json.dumps({'id':doc_id, 'text':text, 'indices':keep,
                              'words':[words[i][:2]+words[i][4:] for i in keep],
                              'token_ranges':{name:[positions[i] for i in keep] for name,positions in loc.items()},
                              'token_counts':sizes}, separators=(',',':')), file=handle)
    RESULTS.mkdir(parents=True, exist_ok=True)
    dump(RESULTS/'common-support.json', {'protocol_sha256':sha(PROTOCOL),
         'original_packet_sha256':meta['packet_sha256'], 'packet_sha256':sha(output),
         'tokenizer_sha256':{name:specs[name]['sha256']['tokenizer.json'] for name in CANDIDATES},
         'documents':support})
    print(sha(output),len(support),sum(d.get('accepted_words',0) for d in support.values()),flush=True)


if __name__ == '__main__':
    main()
