"""Reduce one causal Qwen document forward to sentence features."""

import numpy as np


FEATURE_NAMES = {
    "surprisal": ("final_surprisal_mean_nats", "final_surprisal_max_nats"),
    "intermediate": ("layer12_surprisal_mean_nats", "layer12_entropy_mean_nats"),
    "attention": ("mass_before_sentence", "mass_recent_8_tokens", "normalized_entropy"),
}

_VOCAB_CHUNK_SIZE = 32
_FUTURE_MASS_TOLERANCE = 1e-6


def _token_statistics(torch, lm_head, ids, hidden, norm=None):
    """Use preceding rows for surprisal and same-position rows for entropy."""
    last_query = len(ids) if norm is not None else len(ids) - 1
    surprisal = hidden.new_empty((len(ids) - 1,), dtype=torch.float32)
    entropy = hidden.new_empty((len(ids),), dtype=torch.float32) if norm is not None else None

    for start in range(0, last_query, _VOCAB_CHUNK_SIZE):
        stop = min(start + _VOCAB_CHUNK_SIZE, last_query)
        states = hidden[start:stop]
        logits = lm_head(states if norm is None else norm(states)).float()
        log_normalizer = torch.logsumexp(logits, dim=-1)
        target_ids = ids[start + 1:stop + 1]
        query_count = len(target_ids)
        target_logits = logits[:query_count].gather(1, target_ids[:, None]).squeeze(1)
        surprisal[start:start + query_count] = log_normalizer[:query_count] - target_logits

        if entropy is not None:
            log_probabilities = logits.sub_(log_normalizer[:, None])
            entropy[start:stop] = -log_probabilities.exp().mul_(log_probabilities).sum(dim=-1)

    return surprisal.cpu().numpy(), None if entropy is None else entropy.cpu().numpy()


def _attention_features(torch, attentions, starts, ends, token_count):
    """Summarize sentence-end query rows within the visible document prefix."""
    positions = torch.arange(token_count, device=ends.device)
    visible = positions <= ends[:, None]
    rows = torch.stack([layer.index_select(1, ends).float() for layer in attentions])
    prefix = rows.masked_fill(~visible, 0)
    prefix_total = prefix.sum(dim=-1, keepdim=True)
    future_mass = rows.masked_fill(visible, 0).abs().sum(dim=-1)
    invalid = (~torch.isfinite(rows) | (rows < 0)).any() | (prefix_total <= 0).any()
    invalid = invalid | (future_mass > _FUTURE_MASS_TOLERANCE).any()
    if invalid.item():
        raise ValueError("Invalid attention probabilities or noncausal future attention mass")

    probabilities = prefix / prefix_total
    before_sentence = (probabilities * (positions < starts[:, None])).sum(dim=-1)
    recent = (probabilities * (positions > ends[:, None] - 8)).sum(dim=-1)
    # A one-token prefix has zero entropy and uses denominator one by convention.
    visible_count = (ends + 1).to(torch.float32)
    log_visible_count = visible_count.log().masked_fill(ends == 0, 1)
    entropy = torch.special.entr(probabilities).sum(dim=-1) / log_visible_count
    return torch.stack((before_sentence, recent, entropy), dim=-1).mean(dim=(0, 1)).cpu().numpy()


def signals_for_document(torch, model, token_ids, final_hidden, intermediate_hidden, attentions, sentences):
    """Return float32 arrays in sentence order and the column order of FEATURE_NAMES.

    Hidden states have shape [tokens, width]. The final states already have
    RMSNorm applied. The layer-12 states are taken before the final RMSNorm.
    The six eager full-attention tensors have shape [heads, tokens, tokens].
    All three inputs exclude padding and come from the same backbone forward.
    Each sentence is an ID and an ordered list of positions in token_ids.

    Surprisal is negative log probability in nats for the observed token.
    Token zero has no predecessor and is omitted. A sentence containing only
    token zero has missing surprisal (NaN). Layer-12 entropy is in nats for
    the next-token distribution after reading each token in the sentence.

    Attention statistics average over heads and full-attention layers after
    normalizing each query row on keys up through the sentence end. The
    recent window includes the sentence-end token. Attention entropy uses
    natural logs and is divided by log of the number of visible keys.
    """
    token_count = len(token_ids)
    if (
        token_count == 0
        or final_hidden.ndim != 2
        or final_hidden.shape[0] != token_count
        or intermediate_hidden.shape != final_hidden.shape
    ):
        raise ValueError("Document tokens and hidden states must have matching lengths and widths")
    if (
        len(attentions) != 6
        or attentions[0].ndim != 3
        or attentions[0].shape[0] == 0
        or attentions[0].shape[1:] != (token_count, token_count)
        or any(layer.shape != attentions[0].shape for layer in attentions)
    ):
        raise ValueError("Expected six full-attention tensors over real document tokens")
    if not sentences:
        raise ValueError("A document must contain at least one annotated sentence")

    previous_end = -1
    for sentence_id, positions in sentences:
        if (
            not positions
            or not previous_end < positions[0] <= positions[-1] < token_count
            or any(left >= right for left, right in zip(positions, positions[1:]))
        ):
            raise ValueError(f"Invalid or overlapping token positions for sentence {sentence_id}")
        previous_end = positions[-1]

    device = final_hidden.device
    with torch.inference_mode():
        starts = torch.tensor([positions[0] for _, positions in sentences], device=device, dtype=torch.long)
        ends = torch.tensor([positions[-1] for _, positions in sentences], device=device, dtype=torch.long)
        attention = _attention_features(torch, attentions, starts, ends, token_count)
        ids = torch.tensor(token_ids, device=device, dtype=torch.long)
        final_surprisal, _ = _token_statistics(torch, model.lm_head, ids, final_hidden)
        mid_surprisal, mid_entropy = _token_statistics(
            torch, model.lm_head, ids, intermediate_hidden, norm=model.model.norm
        )

    if not all(np.isfinite(values).all() for values in (final_surprisal, mid_surprisal, mid_entropy)):
        raise ValueError("Nonfinite values from the language-model vocabulary projection")

    surprisal = np.full((len(sentences), 2), np.nan, dtype=np.float32)
    intermediate = np.full_like(surprisal, np.nan)
    for row, (_, positions) in enumerate(sentences):
        scored = [position - 1 for position in positions if position > 0]
        if scored:
            surprisal[row] = (final_surprisal[scored].mean(), final_surprisal[scored].max())
            intermediate[row, 0] = mid_surprisal[scored].mean()
        intermediate[row, 1] = mid_entropy[positions].mean()

    return {"surprisal": surprisal, "intermediate": intermediate, "attention": attention}
