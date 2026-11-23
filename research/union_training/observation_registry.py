"""Approved observation models and the shared states each may read."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Contract:
    likelihood: str
    coordinates: str
    states: tuple[str, ...]


def contracts():
    entries = {}
    for likelihood, coordinates, states, names in [
        ('ordinal', 'target_and_document_end', ('processing',),
         ('plaba_processing_ease',)),
        ('ordinal', 'target_and_document_end', ('interpretation',),
         ('plaba_perceived_understanding',)),
        ('categorical', 'target_and_document_end', ('processing', 'interpretation'),
         ('plaba_issue_set',)),
        ('ordinal', 'point', ('processing',),
         ('plaba_overall_processing_ease', 'plaba_quiz_effort', 'studentannotations_comprehension_difficulty')),
        ('ordinal', 'point', ('interpretation',),
         ('plaba_overall_perceived_understanding', 'plaba_quiz_confidence', 'kerwer_comprehensibility',
          'kerwer_confusion')),
        ('ordinal', 'point', ('appraisal',), ('kerwer_frustration',)),
        ('gaussian', 'target_and_document_end', ('lexical',), ('complex_word_difficulty',)),
        ('categorical', 'target_and_document_end', ('lexical',), ('cwi_native_marked', 'cwi_nonnative_marked')),
        ('hurdle_ordinal', 'target_and_document_end', ('interpretation',),
         ('scarecrow_retrospective_contradiction_mark_ordinal',)),
        ('known_error_gaussian', 'point', ('processing',), ('clear_excerpt_easiness',)),
        ('ordered_choice', 'pair', ('processing',), ('turkish_sentence_comparison',)),
        ('gaussian', 'point', ('processing',),
         ('yadav_full_sentence_difficulty', 'yadav_paced_words_difficulty',
          'textcomplexityde_complexity', 'frick_log_keypress_seconds')),
        ('gaussian', 'point', ('lexical',), ('textcomplexityde_lexical_difficulty',)),
        ('gaussian', 'point', ('interpretation',), ('textcomplexityde_understandability',)),
        ('gaussian', 'point', ('processing', 'interpretation'), ('blott_2023_sense_response_log_milliseconds',)),
        ('categorical', 'point', ('interpretation',),
         ('blott_2021_sentence_end_sense', 'blott_2023_sentence_end_sense',
          'currie_end_coherence_report', 'czech_exp2_qcor', 'czech_exp2_qmis', 'czech_exp2_qpos',
          'czech_exp3_qcor', 'czech_exp3_qmis', 'czech_exp3_qpos', 'first_readback_end_acceptability',
          'frick_after_first_report', 'zhan_reference_form', 'zhan_remention')),
        ('ordinal_binary', 'point', ('interpretation',), ('ambistory_joint',)),
        ('categorical', 'point', ('processing', 'interpretation'), ('context_first_readback', 'first_readback_risk')),
        ('first_event', 'trajectory', ('interpretation',), ('frick_first_report',)),
        ('categorical', 'point', ('retrieval',),
         ('oliver_retest_initial_state', 'oliver_retest_target_confirmation', 'oliver_retrieval_protocol_path')),
        ('gaussian', 'point', ('retrieval',),
         ('oliver_first_confirmation_rt_log_milliseconds', 'oliver_first_definition_rt_log_milliseconds',
          'oliver_first_delay_rt_log_milliseconds', 'oliver_second_confirmation_rt_log_milliseconds',
          'oliver_second_definition_rt_log_milliseconds')),
        ('choice', 'point', ('interpretation',), ('onestop_answer_choice', 'race_answer_choice')),
        ('categorical', 'point', ('interpretation',),
         ('onestop_rereading_first_read_correct', 'onestop_rereading_reread_correct',
          'plaba_open_book_accuracy', 'turkish_comprehension_correct')),
        ('paired_gaussian', 'pair', ('belief',), ('rodger_paired_effectiveness', 'rodger_paired_support')),
        ('gaussian', 'point', ('belief',),
         ('qiao_confidence', 'qiao_evidence_likelihood_if_guilty',
          'qiao_evidence_likelihood_if_not_guilty', 'qiao_guilt_rating')),
        ('categorical', 'point', ('belief',), ('qiao_final_verdict',)),
    ]:
        for name in names:
            if name in entries:
                raise ValueError(f'Duplicate observation contract: {name}')
            entries[name] = Contract(likelihood, coordinates, states)
    return entries


REPLACED = frozenset({
    'ambistory_given_sense_nonsensical_flag', 'ambistory_given_sense_plausibility',
    'frick_first_report_hazard', 'frick_sentence_inconsistent', 'frick_target_first_report',
    'onestop_answer_accuracy', 'race_answer_accuracy',
    'plaba_issue_awkward_wordy_vague', 'plaba_issue_other', 'plaba_issue_report',
    'plaba_issue_too_dense', 'plaba_issue_unclear_reference', 'plaba_issue_unclear_takeaway',
    'plaba_issue_unclear_term', 'rodger_change_effectiveness', 'rodger_change_support',
    'rodger_rating_effectiveness', 'rodger_rating_support', 'turkish_earlier_sentence_easier',
})

RECONSTRUCTED = {
    'ambistory_joint': ('ambistory', tuple(f'{level}|{flag}' for level in range(1, 6) for flag in (0, 1))),
    'frick_first_report': ('frick', ()),
    'frick_after_first_report': ('frick', (0, 1)),
    'plaba_issue_set': ('plaba', ()),
    'rodger_paired_effectiveness': ('rodger', ('before', 'after')),
    'rodger_paired_support': ('rodger', ('before', 'after')),
    'turkish_sentence_comparison': ('turkish', (0, 1)),
}
