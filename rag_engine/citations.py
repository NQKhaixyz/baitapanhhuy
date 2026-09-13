"""Citation formatting/whitelisting is mechanical, not semantic verification."""
from __future__ import annotations
import logging
import re
from .guardrails import REFUSAL, is_pure_refusal, is_refusal_clause

LOGGER = logging.getLogger('mini_rag.cite')


def citation_ids(answer: str) -> list[str]:
    return re.findall(r"\[([^\]]+)\]", answer)


def claims_have_citations(answer: str) -> bool:
    if is_pure_refusal(answer):
        return True
    # A bullet may contain several source-backed clauses separated by ``;``
    # and put one citation at the end (the format produced by the prompt).
    # Keep semicolons inside the same claim; sentence boundaries still split
    # when the next sentence does not begin with its own citation.
    segments = re.split(r"\n+|(?<=[.!?])\s+(?!\[)", answer)
    for i, segment in enumerate(segments):
        content = re.sub(r"\[[^\]]+\]", "", segment).strip(" -•\t`")
        if not content or citation_ids(segment) or is_refusal_clause(content):
            continue
        if content.endswith(':') and citation_ids(' '.join(segments[i+1:i+3])):
            continue
        return False
    return True


def answer_is_grounded(answer: str, hits: list[dict]) -> bool:
    cited = citation_ids(answer)
    allowed = {h['chunk_id'] for h in hits}
    if is_pure_refusal(answer):
        # A canonical refusal is citation-free. If a model appended IDs, still
        # enforce the whitelist so an invented ID cannot cross the cite node.
        return set(cited).issubset(allowed)
    return bool(cited) and set(cited).issubset(allowed) and claims_have_citations(answer)


def cite_answer(answer: str, hits: list[dict]) -> dict:
    """Final node: reject invalid IDs; never manufacture citations for claims."""
    valid = answer_is_grounded(answer, hits)
    final = answer if valid else REFUSAL
    result = {'answer': final, 'citations': citation_ids(final), 'citation_contract_ok': valid}
    LOGGER.info('cite input=%r allowed_ids=%s -> %s', answer, [h['chunk_id'] for h in hits], result)
    return result
