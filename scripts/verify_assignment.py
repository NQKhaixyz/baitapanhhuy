"""Exercise all stage CLIs plus conversations; retain evidence without editing gold.

Run from the repo: python scripts/verify_assignment.py --offline
Omit --offline to use the configured embedding/generation APIs.
Generation correctness still requires reading the answers, not this script's flags.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

# Windows may start Python with a legacy cp1252 console even when child
# processes receive PYTHONIOENCODING=utf-8 below.  The harness prints corpus
# questions (which include Vietnamese), so configure this process as well.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--output',default='artifacts/verification_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    args=parser.parse_args()
    from rag_engine import config
    env=os.environ.copy()
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    env['PYTHONIOENCODING']='utf-8'
    env['GEMINI_RETRY_ATTEMPTS']='3'
    env['GEMINI_REQUEST_INTERVAL_SECONDS']='6'
    env['RAG_GENERATION_CACHE_DIR']=str(out/'generation')
    env['RAG_EMBEDDING_CACHE']=str(out/'corpus_cli.npy')
    config.DEFAULT_EMBEDDING_CACHE=out/'corpus_cli.npy'
    if args.offline:
        env.update(RAG_OFFLINE='1',RAG_EMBEDDING_PROVIDER='local',GEMINI_API_KEY='',OPENAI_API_KEY='',RAG_RETRIEVAL_THRESHOLD='0.18')
    os.environ.update(env)
    from rag_engine.graph import MiniRAGGraph
    from rag_engine.retrieval import Retriever
    from rag_engine.guardrails import is_pure_refusal
    from rag_engine.synthesis import GeminiAnswerGenerator
    from rag_engine.io_utils import read_jsonl, write_json
    from rag_engine.core import retrieval_metrics
    threshold=float(env.get('RAG_RETRIEVAL_THRESHOLD','0.18'))
    mode='offline' if args.offline else 'live'
    commands=[['stage0.py','--all'],['stage1.py','--all'],['stage2.py','--all'],['stage3.py'],
              ['stage4.py','--output',str(out/'standard_k3.json')],
              ['eval.py','--k','1','--output',str(out/'standard_k1.json')],
              ['eval.py','--prompt-variant','concise','--output',str(out/'concise_k3.json')]]
    summary={'mode':mode,'commands':[],'conversations':[]}
    def save(): write_json(out/'verification.json',summary)
    for command in commands:
        label='stage'+str(len(summary['commands']))
        print('RUN '+' '.join(command),flush=True)
        result=subprocess.run([sys.executable,*command],cwd=ROOT,env=env,capture_output=True,
                              text=True,encoding='utf-8',errors='replace',timeout=240)
        (out/(label+'.log')).write_text(result.stdout+'\nSTDERR:\n'+result.stderr,encoding='utf-8')
        summary['commands'].append({'command':command,'exit_code':result.returncode})
        save()
        if result.returncode: raise RuntimeError('CLI failed: '+str(command))
    from eval import compare_reports
    read=lambda name:json.loads((out/name).read_text(encoding='utf-8'))
    write_json(out/'prompt_comparison.json',compare_reports(read('standard_k3.json'),read('concise_k3.json')))
    write_json(out/'k_comparison.json',compare_reports(read('standard_k3.json'),read('standard_k1.json')))
    retriever=Retriever(cache_path=out/'corpus.npy')
    baseline=[{**q,'predicted_chunk_ids':[h['chunk_id'] for h in retriever.search(q['question'],3)]}
              for q in read_jsonl(config.DEFAULT_QA_PATH)]
    summary['baseline_metrics']=retrieval_metrics(baseline)
    summary['retrieval_did_not_regress']=(read('standard_k3.json')['metrics']['recall_at_3'] >=
                                         summary['baseline_metrics']['recall_at_3'])

    class TracedGenerator(GeminiAnswerGenerator):
        def __init__(self): super().__init__(); self.raw=[]
        def generate(self,prompt):
            answer=super().generate(prompt)
            self.raw.append(answer)
            return answer

    sequences=[
        ['xin chào','thời tiết hôm nay thế nào','Giá vàng tăng cao phải làm gì?'],
        ['Xin chào, metformin có chống chỉ định gì?'],
        ['Tôi bị đau răng'],
        ['Tăng huyết áp là gì?','Thế phân độ ra sao?'],
        ['Metformin dùng khi nào?','Còn chống chỉ định?','Vậy eGFR 20 thì sao?'],
        ['Tăng huyết áp là gì?','Còn đái tháo đường cần xét nghiệm gì?'],
        ['Liều metformin khởi đầu là bao nhiêu mg mỗi ngày?'],
        ['Metformin có tác dụng phụ nào?'],
    ]
    for questions in sequences:
        generator=None if args.offline else TracedGenerator()
        graph=MiniRAGGraph(retriever,generator=generator,threshold=threshold)
        states=[]
        for question in questions:
            print('QUERY '+question,flush=True)
            start=len(generator.raw) if generator else 0
            state=graph.invoke(question)
            state.pop('query_vector',None)
            if generator: state['raw_model_responses']=generator.raw[start:]
            states.append(state)
        summary['conversations'].append(states)
        save()
    c=summary['conversations']
    summary['regression_checks']={
        'nonmedical_no_search':all('hits' not in s for s in c[0]),
        'greeting_with_question_is_medical':c[1][0]['category']=='medical',
        'medical_outside_corpus_is_medical':c[2][0]['category']=='medical',
        'classification_followup_hit':'tha2022_ch2_b2.2' in [h['chunk_id'] for h in c[3][-1]['hits']],
        'third_turn_keeps_drug':'metformin' in c[4][-1]['standalone_query'].lower(),
        'topic_switch_filters_old_guideline':c[5][-1]['guideline_ids']==['dtd2020'],
    }
    if not args.offline:
        summary['regression_checks'].update({
            'dosage_refusal_preserved':is_pure_refusal(c[6][0]['answer']) and c[6][0]['rejected'],
            'adverse_effect_refusal_preserved':is_pure_refusal(c[7][0]['answer']) and c[7][0]['rejected'],
            'topic_switch_answered':not c[5][-1]['rejected'],
        })
    save()
    if not summary['retrieval_did_not_regress'] or not all(summary['regression_checks'].values()):
        raise RuntimeError('Regression failed; inspect verification.json')
    print(json.dumps(summary['regression_checks'],ensure_ascii=False,indent=2))


if __name__=='__main__': main()
