"""Mechanical checks aid, but never replace, manual answer review."""
from __future__ import annotations
from typing import Any
from .metrics import gold_chunk_ids, retrieval_metrics
from .guardrails import content_terms, numeric_terms, is_pure_refusal, refusal_text, numeric_violations
from .citations import citation_ids, claims_have_citations
from .synthesis import answer_violations


def reference_terms_present(item: dict, answer: str) -> bool:
    """Surface overlap only. This flag deliberately makes NO semantic claim."""
    terms, numbers = content_terms(answer), numeric_terms(answer)
    for group in item.get('retrieval_gt', []):
        for point in group.get('points', []):
            text = str(point.get('text') or point.get('verbatim_quote') or '')
            required = numeric_terms(text)
            if required and not required.issubset(numbers):
                return False
            words = content_terms(text)
            if words and len(words & terms) < min(2, len(words)):
                return False
    return True


def answer_checks(item: dict[str, Any], answer: str, hits: list[dict],
                  generation_status: str | None = None) -> dict:
    answerable = bool(item.get('meta', {}).get('is_answerable', True))
    cited = citation_ids(answer)
    allowed = {h['chunk_id'] for h in hits}
    errors = answer_violations(item.get('question', ''), answer, hits)
    valid = set(cited).issubset(allowed) and claims_have_citations(answer)
    if not answerable:
        return {'answerable': False, 'correct_refusal': is_pure_refusal(answer) and valid,
                'citation_ids_valid': valid, 'violations': errors}
    valid = valid and bool(cited)
    gold_ids = gold_chunk_ids(item)
    if len(gold_ids) > 1:
        valid = valid and gold_ids.issubset(set(cited))
    texts = {h['chunk_id']: str(h['chunk'].get('text', '')) for h in hits}
    quotes = all(point.get('verbatim_quote', '') in ' '.join(texts.get(i, '') for i in group.get('chunk_ids', []))
                 for group in item.get('retrieval_gt', []) for point in group.get('points', []))
    surface = reference_terms_present(item, answer)
    mechanical = (surface and quotes and valid and not errors and not refusal_text(answer)
                  and generation_status not in {'refused', 'fallback'})
    return {'answerable': True, 'reference_terms_present': surface,
            'verbatim_quotes_in_context': quotes, 'citations_present': bool(cited),
            'citation_coverage_ok': claims_have_citations(answer), 'citation_ids_valid': valid,
            'numeric_consistency_ok': not numeric_violations(item.get('question', ''), answer, hits),
            'violations': errors, 'mechanical_checks_pass': bool(mechanical),
            'answer_correct': None, 'manual_review_required': True,
            'answer_gt_reference': item.get('answer_gt', '')}
