"""Bind the live Gradio verification and published holdout scores to one candidate."""

import argparse
import json
from pathlib import Path

from .causal_plaba import digest
from .checkpoint import DEFAULT_CHECKPOINT, load_checkpoint, read


RESULTS = Path(__file__).resolve().parents[1] / 'results'


def check_members(directory, members, names):
    for source, local in names.items():
        if digest(directory / local) != members[source]:
            raise ValueError(f'Saved verification artifact differs from its Colab stage receipt: {local}')


def check_forecast(directory, project, candidate, name, count):
    local = directory / name
    score, seal = (read(local / path) for path in ('scores.json', 'seal.json'))
    spelling = 'forecast' if name == 'blott' else 'forecasts'
    protocol = project / f'{name}-transfer-protocol.json'
    prereg = read(protocol)
    input_key = prereg['input_sha256']['stimuli']
    if (score['candidate_id'] != candidate or seal['candidate_id'] != candidate or prereg['candidate_id'] != candidate
            or seal['input_sha256'] != input_key or seal['documents'] != count or seal['forwards'] != count
            or seal['predictions_sha256'] != digest(local / 'predictions.jsonl')
            or score[f'{spelling}_sha256'] != seal['predictions_sha256']
            or score[f'{spelling}_seal_sha256'] != digest(local / 'seal.json')
            or score['protocol_sha256'] != digest(protocol)):
        raise ValueError(f'{name} results are not attributable to the fixed candidate and locked protocol')
    return score


def check_medreadme(directory, checkpoint):
    run, assay, seal, score, runtime = (read(directory / name) for name in (
        'run-verification.json', 'assay.json', 'seal.json', 'scores.json', 'runtime.json'))
    stages = run['stages']
    if (run['phase'] != 'finished' or not run['release_verified'] or not run['endpoint_absent_after_stop']
            or set(stages) != {'setup', 'medreadme'} or runtime['device'] != 'NVIDIA L4'
            or any(row['verified'] is not True or row['state'] != 'complete' for row in stages.values())):
        raise ValueError('Expected the released MedReadMe L4 verification')
    check_members(directory, stages['setup']['members'], {'runtime.json': 'runtime.json'})
    check_members(directory, stages['medreadme']['members'], {'predictions.jsonl': 'predictions.jsonl', 'seal.json': 'seal.json'})
    stage6 = Path(__file__).resolve().parents[1] / 'stage6'
    if (any(row['candidate_id'] != checkpoint.identifier for row in (assay, seal, score))
            or assay['protocol_sha256'] != digest(stage6 / 'medreadme-lexical-holdout.md')
            or seal['forecast_code_sha256'] != digest(stage6 / 'medreadme_lexical_forecast.py')
            or score['scoring_code_sha256'] != digest(stage6 / 'medreadme_lexical_score.py')
            or seal['lexical_weights_sha256'] != assay['lexical_weights_sha256']
            or seal['lexical_weights_sha256'] != digest(checkpoint.directory / 'lexical/heads.npz')
            or seal['stimuli_sha256'] != assay['stimuli_sha256'] or seal['model_forwards'] != assay['retained_sentences']
            or seal['model_forwards'] != score['sentences'] or seal['forecasts_sha256'] != digest(directory / 'predictions.jsonl')
            or score['assay_sha256'] != digest(directory / 'assay.json') or score['forecast_seal_sha256'] != digest(directory / 'seal.json')
            or run['scores_sha256'] != digest(directory / 'scores.json')):
        raise ValueError('MedReadMe score differs from the candidate, input or sealed forecasts')
    acceptance = score['aligned_fraction'] >= 0.95 and all(
        row['paired_95_ci'][0] > 0 for row in score['paired_served_minus_baseline'].values())
    if set(score['paired_served_minus_baseline']) != {'length', 'inverse_frequency'} or score['passed'] != acceptance:
        raise ValueError('MedReadMe acceptance does not match its fixed criteria')
    return {'passed': acceptance, 'sentences': score['sentences'], 'sources': score['sources'],
            'eligible_word_coverage': score['aligned_fraction'],
            'mean_within_sentence_average_precision': score['mean_within_sentence_average_precision'],
            'scores_sha256': run['scores_sha256']}


def verify(checkpoint_dir, project):
    checkpoint = load_checkpoint(checkpoint_dir)
    directory = project / 'gradio-candidate-l4'
    run, gradio = (read(directory / path) for path in ('run-verification.json', 'verification.json'))
    runtime = read(directory / 'runtime.json')
    stages = run['stages']
    if (run['checkpoint_sha256'] != checkpoint.identifier or gradio['checkpoint_sha256'] != checkpoint.identifier
            or run['phase'] != 'finished' or run['release_verified'] is not True
            or run['endpoint_absent_after_stop'] is not True or set(stages) != {'setup', 'gradio', 'forecast'}
            or any(row['verified'] is not True or row['state'] != 'complete' for row in stages.values())
            or runtime['device'] != 'NVIDIA L4' or gradio['device'] != 'NVIDIA L4 FP32'
            or gradio['model_forwards_per_combined_gradio_request'] != 1
            or gradio['gradio_cached_views_verified'] is not True or gradio['model_forwards_for_three_cached_interactions'] != 0):
        raise ValueError('Expected the released L4 runtime and single-forward Gradio verification')
    check_members(directory, stages['setup']['members'], {'runtime.json': 'runtime.json'})
    check_members(directory, stages['gradio']['members'], {
        'verification.json': 'verification.json', 'serving-seal.json': 'serving-seal.json',
        'replay-report.json': 'replay-report.json', 'predictions.jsonl': 'iirc-predictions.jsonl',
    })
    check_members(directory, stages['forecast']['members'], {f'{name}/{file}': f'{name}/{file}'
                  for name in ('blott', 'kerwer') for file in ('seal.json', 'predictions.jsonl')})
    for name, path in (('plaba', 'sentence/head.npz'), ('ordinal', 'ordinal/head.npz'), ('lexical', 'lexical/heads.npz'),
                       ('iirc', 'anchor/heads.npz')):
        if gradio[f'{name}_head_sha256'] != digest(checkpoint_dir / path):
            raise ValueError(f'Serving verification uses different {name} weights')

    anchors = read(directory / 'iirc-locked-scores.json')
    lexical = read(checkpoint_dir / 'lexical/replay.json')
    if (anchors['input_sha256']['heads'] != gradio['iirc_head_sha256'] or anchors['paragraphs'] != 430
            or anchors['input_sha256']['seal'] != digest(directory / 'serving-seal.json')
            or anchors['input_sha256']['predictions'] != digest(directory / 'iirc-predictions.jsonl')
            or not anchors['acceptance']['all_conditions_passed']
            or lexical['maximum_frozen_prediction_difference'] > lexical['score_tolerance']):
        raise ValueError('Expected the original anchor acceptance and target-word numerical replay')
    blott = check_forecast(directory, project, checkpoint.identifier, 'blott', 188)
    kerwer = check_forecast(directory, project, checkpoint.identifier, 'kerwer', 36)
    medreadme = check_medreadme(project / 'medreadme-lexical-l4', checkpoint)
    blott_rank = blott['rank_agreement']['within_quartet']['understanding_minus_shorter_sentence']
    return {
        'candidate_id': checkpoint.identifier, 'run_sha256': digest(directory / 'run-verification.json'),
        'artifact_hashes': {path: digest(directory / path) for path in ('verification.json', 'iirc-locked-scores.json',
                                                                         'blott/scores.json', 'kerwer/scores.json')},
        'execution': {'device': gradio['device'], 'forwards_per_analysis': 1, 'cached_interaction_forwards': 0},
        'evidence': {
            'annotator_selected_spans_after_reading_iirc_paragraph': {
                'passed': anchors['acceptance']['all_conditions_passed'], 'paragraphs': anchors['paragraphs'],
                'both_same_stop_targets': anchors['distinct_same_stop_pair_coverage']['means'],
            },
            'designated_complex_lexical_target_replay': {
                'passed_numeric_replay': True, 'designated_targets': lexical['replay_count'],
                'mae_0_to_1': lexical['mae_0_to_1'],
            },
            'english_medical_difficult_word_localization': medreadme,
            'article_comprehensibility_rewrite_kerwer': {
                'passed_direction_and_magnitude': kerwer['primary']['effect_gate_passes'],
                'improved_absolute_error_over_length': kerwer['primary']['improves_on_length'],
            },
            'sentence_coherence_rank_blott': {'improved_over_length': blott_rank['ci95'][0] > 0, **blott_rank},
        },
        'directly_unverified_candidate_outputs': ['reader_background', 'question_onset', 'question_resolution',
                                                   'independent_sentence_rating_distribution_calibration'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--results', type=Path, default=RESULTS)
    args = parser.parse_args()
    print(json.dumps(verify(args.checkpoint, args.results), indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
