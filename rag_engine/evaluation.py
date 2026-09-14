"""Mechanical checks aid, but never replace, manual answer review."""
from __future__ import annotations
from typing import Any
from .core import gold_chunk_ids
from .guardrails import content_terms, numeric_terms, is_pure_refusal, refusal_text, numeric_violations
from .core import citation_ids, claims_have_citations
from .synthesis import answer_violations
from .embeddings import normalize_text


def reference_terms_present(item: dict, answer: str) -> bool:
    """Surface overlap only. This flag deliberately makes NO semantic claim."""
    terms, numbers = content_terms(answer), numeric_terms(answer)
    point_numbers: set[str] = set()
    for group in item.get('retrieval_gt', []):
        for point in group.get('points', []):
            text = str(point.get('text') or point.get('verbatim_quote') or '')
            point_numbers.update(numeric_terms(text))
    reference_numbers = numeric_terms(str(item.get('answer_gt', '')))
    # A number is mechanically required only when the reference uses it and
    # it is anchored in the question/evidence. This excludes prose counts such
    # as "đạt 1 trong các ngưỡng" while retaining patient values and thresholds.
    reference_numbers &= point_numbers | numeric_terms(str(item.get('question', '')))
    if reference_numbers and not reference_numbers.issubset(numbers):
        return False
    for group in item.get('retrieval_gt', []):
        for point in group.get('points', []):
            text = str(point.get('text') or point.get('verbatim_quote') or '')
            required = numeric_terms(text) if not item.get('answer_gt') else set()
            if required and not required.issubset(numbers):
                return False
            point_numeric = numeric_terms(text)
            words = {word for word in content_terms(text)
                     if word.replace(',', '.') not in point_numeric}
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
    gold_citations_ok = not gold_ids or gold_ids.issubset(set(cited))
    texts = {h['chunk_id']: str(h['chunk'].get('text', '')) for h in hits}
    quotes = all(point.get('verbatim_quote', '') in ' '.join(texts.get(i, '') for i in group.get('chunk_ids', []))
                 for group in item.get('retrieval_gt', []) for point in group.get('points', []))
    surface = reference_terms_present(item, answer)
    mechanical = (surface and quotes and valid and gold_citations_ok
                  and not errors and not refusal_text(answer)
                  and generation_status not in {'refused', 'fallback'})
    return {'answerable': True, 'reference_terms_present': surface,
            'verbatim_quotes_in_context': quotes, 'citations_present': bool(cited),
            'citation_coverage_ok': claims_have_citations(answer), 'citation_ids_valid': valid,
            'gold_citations_ok': gold_citations_ok,
            'numeric_consistency_ok': not numeric_violations(item.get('question', ''), answer, hits),
            'violations': errors, 'mechanical_checks_pass': bool(mechanical),
            'answer_correct': None, 'manual_review_required': True,
            'answer_gt_reference': item.get('answer_gt', '')}


def probe_answer_checks(item: dict[str, Any], answer: str, hits: list[dict],
                        generation_status: str | None = None) -> dict[str, Any] | None:
    """Evaluate an external alias/probe row with explicit expected signals.

    This is intentionally separate from the golden mechanical checks. The
    probe file can state the minimum observable evidence for a paraphrased
    question without changing the read-only golden dataset. It reports an
    auditable contract, not a claim of full semantic understanding.
    """
    expected = item.get('expected')
    if not isinstance(expected, dict):
        return None
    normalized = normalize_text(answer)
    answerable = bool(expected.get('answerable', True))
    refused = is_pure_refusal(answer) or generation_status in {'refused', 'fallback'}
    required = [normalize_text(str(term)) for term in expected.get('must_include', [])]
    alternatives = [
        [normalize_text(str(term)) for term in group]
        for group in expected.get('must_include_any', [])
        if group
    ]
    forbidden = [normalize_text(str(term)) for term in expected.get('must_not_include', [])]
    missing = [term for term in required if term not in normalized]
    missing_alternatives = [group for group in alternatives if not any(term in normalized for term in group)]
    forbidden_present = [term for term in forbidden if term in normalized]
    cited = citation_ids(answer)
    allowed = {hit['chunk_id'] for hit in hits}
    gold = gold_chunk_ids(item)
    citation_contract_ok = set(cited).issubset(allowed) and claims_have_citations(answer)
    if answerable:
        citation_contract_ok = citation_contract_ok and bool(cited)
    gold_citations_ok = not gold or gold.issubset(set(cited))
    grounding_errors = answer_violations(item.get('question', ''), answer, hits)
    if answerable:
        passed = (not refused and not missing and not missing_alternatives
                  and not forbidden_present and citation_contract_ok
                  and gold_citations_ok and not grounding_errors)
    else:
        passed = refused and citation_contract_ok
    return {
        'answerable_expected': answerable,
        'answer_correct': bool(passed),
        'false_refusal': bool(answerable and refused),
        'missing_terms': missing,
        'missing_alternatives': missing_alternatives,
        'forbidden_terms_present': forbidden_present,
        'citation_contract_ok': citation_contract_ok,
        'gold_citations_ok': gold_citations_ok,
        'violations': grounding_errors,
        'generation_status': generation_status,
    }
