"""Run retrieval eval, archive reproducible reports, and attach manual review."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
from rag_engine.config import (ROOT_DIR, DEFAULT_CORPUS_PATH, DEFAULT_QA_PATH,
    DEFAULT_EVAL_OUTPUT, RETRIEVAL_THRESHOLD, TOP_K, MIN_QUERY_COVERAGE, ROUTE_MARGIN)
from rag_engine.evaluation import answer_checks
from rag_engine.core import gold_chunk_ids, retrieval_metrics
from rag_engine.graph import MiniRAGGraph
from rag_engine.io_utils import read_jsonl, write_json, ensure_utf8_output
from rag_engine.retrieval import Retriever
from rag_engine.synthesis import GROUNDING_INSTRUCTION, PROMPT_VARIANTS


def digest(value) -> str:
    payload = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def run_eval(output_path=DEFAULT_EVAL_OUTPUT, *, top_k=TOP_K, threshold=RETRIEVAL_THRESHOLD,
             generator=None, prompt_variant='standard') -> dict:
    if top_k < 1:
        raise ValueError('k phải >= 1')
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError('Unknown prompt variant')
    rows = read_jsonl(DEFAULT_QA_PATH)
    retriever = Retriever()
    graph = MiniRAGGraph(retriever, generator=generator, top_k=top_k, threshold=threshold,
                         enable_rewrite=False, prompt_variant=prompt_variant)
    results = []
    for row in rows:
        state = graph.invoke(row['question'])
        hits = state.get('hits', [])
        generation = state.get('generation', {'status':'not_called'})
        results.append({
            'id': row['id'], 'question': row['question'], 'gold_chunk_ids': sorted(gold_chunk_ids(row)),
            'predicted_chunk_ids': [h['chunk_id'] for h in hits],
            'scores': {h['chunk_id']:h['score'] for h in hits},
            'answer': state['answer'], 'answer_sha256': digest(state['answer']),
            'category': state['category'], 'guideline_ids': state.get('guideline_ids', []),
            'classification_source': state.get('classification_source'),
            'classification_model': state.get('classification_model'),
            'classification_api_calls': state.get('classification_api_calls', 0),
            'rejected': state['rejected'], 'generation': generation,
            'checks': answer_checks(row, state['answer'], hits, generation.get('status')),
        })
    source_files = sorted(list(ROOT_DIR.glob('*.py')) + list((ROOT_DIR/'rag_engine').glob('*.py')))
    code_hash = digest({str(p.relative_to(ROOT_DIR)):digest(p.read_bytes()) for p in source_files})
    answerable = [i for i in results if i['checks']['answerable']]
    summary = {
        'answerable_count': len(answerable), 'unanswerable_count':len(results)-len(answerable),
        'correct_refusal':sum(i['checks'].get('correct_refusal', False) for i in results),
        'mechanical_pass':sum(i['checks'].get('mechanical_checks_pass', False) for i in answerable),
        'verbatim_quote_pass':sum(i['checks'].get('verbatim_quotes_in_context', False) for i in answerable),
        'fallback_count':sum(i['generation']['status']=='fallback' for i in results),
        'manually_reviewed':0, 'manual_correct':None,
    }
    report = {
        'schema_version':2, 'generated_at':datetime.now(timezone.utc).isoformat(),
        'dataset':str(DEFAULT_QA_PATH), 'top_k':top_k, 'retrieval_threshold':threshold,
        'provenance':{
            'code_sha256':code_hash, 'corpus_sha256':digest(DEFAULT_CORPUS_PATH.read_bytes()),
            'qa_sha256':digest(DEFAULT_QA_PATH.read_bytes()),
            'embedding_provider':type(retriever.provider).__name__,
            'embedding_model':getattr(retriever.provider, 'model', None),
            'embedding_dimension':retriever.vectors.shape[1],
            'generation_models':sorted({i['generation']['model'] for i in results if i['generation'].get('model')}),
            'classifier_mode': graph.classifier.mode,
            'classifier_model': graph.classifier.model,
            'prompt_variant':prompt_variant,
            'prompt_sha256':digest(GROUNDING_INSTRUCTION+PROMPT_VARIANTS[prompt_variant]),
            'min_query_coverage':MIN_QUERY_COVERAGE, 'route_margin':ROUTE_MARGIN,
            'generation_parameters':{'temperature':0.2,'max_output_tokens':1000,'candidate_count':1},
            'packages':{name:importlib.metadata.version(name) for name in ('numpy','google-genai','langgraph')},
        },
        'metrics':retrieval_metrics(results), 'answer_summary':summary, 'items':results,
    }
    report['report_id'] = digest(report)
    path = Path(output_path)
    archive = path.with_name(path.stem+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+path.suffix)
    write_json(archive, report)
    write_json(path, report)
    return report


def review_template(report: dict) -> dict:
    return {'report_id':report['report_id'], 'reviewer':'', 'reviewed_at':'',
            'items':[{'id':i['id'], 'answer_sha256':i['answer_sha256'], 'correct':None, 'notes':''}
                     for i in report['items']]}


def attach_review(report: dict, review: dict) -> dict:
    if review.get('report_id') != report['report_id']:
        raise ValueError('Manual review belongs to another report')
    if not review.get('reviewer') or not review.get('reviewed_at'):
        raise ValueError('Review requires reviewer and timestamp')
    items = review.get('items', [])
    reviews = {i['id']:i for i in items}
    if len(reviews) != len(items) or set(reviews) != {i['id'] for i in report['items']}:
        raise ValueError('Review must cover every item exactly once')
    for item in report['items']:
        checked = reviews[item['id']]
        if digest(item['answer']) != item['answer_sha256']:
            raise ValueError('Report answer does not match its stored hash')
        if checked.get('answer_sha256') != item['answer_sha256']:
            raise ValueError('Answer changed after review')
        if type(checked.get('correct')) is not bool or not checked.get('notes', '').strip():
            raise ValueError('Review requires a decision and evidence notes for every answer')
    report['manual_review'] = review
    report['answer_summary']['manually_reviewed'] = len(items)
    report['answer_summary']['manual_correct'] = sum(i['correct'] for i in items)
    return report


def compare_reports(before: dict, after: dict) -> dict:
    metrics = {key:round(after['metrics'][key]-value,4) for key,value in before['metrics'].items()
               if key in after['metrics'] and isinstance(value,(int,float))}
    prior = {i['id']:i for i in before['items']}
    changed = [i['id'] for i in after['items'] if prior.get(i['id'],{}).get('answer') != i['answer']]
    fields = ['code_sha256','corpus_sha256','qa_sha256','embedding_provider','embedding_model',
              'generation_models','classifier_mode','classifier_model',
              'min_query_coverage','route_margin']
    differences = [key for key in fields if before.get('provenance',{}).get(key)!=after.get('provenance',{}).get(key)]
    differences += [key for key in ['top_k','retrieval_threshold'] if before.get(key)!=after.get(key)]
    summaries = {key:after.get('answer_summary',{}).get(key)-value
                 for key,value in before.get('answer_summary',{}).items()
                 if type(value) in (int,float) and type(after.get('answer_summary',{}).get(key)) in (int,float)}
    return {'before':before['report_id'], 'after':after['report_id'], 'metric_deltas':metrics,
            'answer_summary_deltas':summaries,
            'changed_answers':changed, 'other_configuration_changes':differences,
            'prompt_changed':before.get('provenance',{}).get('prompt_sha256') != after.get('provenance',{}).get('prompt_sha256')}


def print_report(report: dict):
    print('ID           recall@1 recall@3 MRR  quote refusal generation')
    for item in report['items']:
        m=retrieval_metrics([item]); c=item['checks']
        print(f"{item['id']:12} {m['recall_at_1']:.2f}     {m['recall_at_3']:.2f}     {m['mrr']:.2f} "
              f"{str(c.get('verbatim_quotes_in_context','-')):5} "
              f"{str(c.get('correct_refusal','-')):7} {item['generation']['status']}")
    print('Retrieval metrics:',report['metrics'])
    print('Mechanical checks (NOT semantic accuracy):',report['answer_summary'])
    print('Answer correctness requires manual review against answer_gt and corpus.')


def main():
    ensure_utf8_output()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('-k','--k',type=int,default=TOP_K)
    parser.add_argument('--threshold',type=float,default=RETRIEVAL_THRESHOLD)
    parser.add_argument('--prompt-variant',choices=tuple(PROMPT_VARIANTS),default='standard')
    parser.add_argument('--write-review-template',action='store_true')
    parser.add_argument('--review-report')
    parser.add_argument('--review-file')
    parser.add_argument('--compare',nargs=2,metavar=('BEFORE','AFTER'))
    args=parser.parse_args()
    if args.k < 1: parser.error('k phải >= 1')
    read=lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
    logging.basicConfig(level=logging.INFO,format='%(levelname)s %(name)s | %(message)s')
    if args.compare:
        result=compare_reports(*(read(p) for p in args.compare))
        print(json.dumps(result,ensure_ascii=False,indent=2))
        if args.output: write_json(args.output,result)
        return
    if args.review_report or args.review_file:
        if not (args.review_report and args.review_file): parser.error('Provide both review paths')
        report=attach_review(read(args.review_report),read(args.review_file))
        write_json(args.output or args.review_report,report)
    else:
        report=run_eval(args.output or DEFAULT_EVAL_OUTPUT, top_k=args.k,
                        threshold=args.threshold,prompt_variant=args.prompt_variant)
        if args.write_review_template:
            path=Path(args.output or DEFAULT_EVAL_OUTPUT).with_suffix('.review.json')
            write_json(path,review_template(report))
    print_report(report)


if __name__=='__main__':
    main()
