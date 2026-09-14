"""Behavioral regressions from the review; no network, no golden modifications."""
import copy
import logging
import subprocess
import sys
import pytest
from rag_engine.config import DEFAULT_QA_PATH
from rag_engine.io_utils import read_jsonl
from rag_engine.retrieval import Retriever
from rag_engine.graph import MiniRAGGraph
from rag_engine.conversation import ConversationMemory
from rag_engine.guardrails import (REFUSAL, numeric_terms, numeric_violations,
                                   question_intent_violations)
from rag_engine.synthesis import (
    GENERATION_PARAMETERS,
    answer_violations,
    generation_cache_key,
    synthesize_answer,
)
from rag_engine.evaluation import (answer_checks, probe_answer_checks,
                                   reference_terms_present)
from rag_engine.core import retrieval_metrics
from eval import attach_review, review_template, compare_reports, digest
from rag_engine.embeddings import (LocalHashEmbedding, get_embedding_provider,
                                   load_or_create_embeddings)
from rag_engine.core import (cite_answer, claims_have_citations,
                             QuestionClassifier)
from web_app import parse_run_request


class Fixed:
    def __init__(self, *answers): self.answers, self.calls = answers, 0
    def generate(self, prompt):
        value = self.answers[min(self.calls, len(self.answers)-1)]
        self.calls += 1
        return value


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv('RAG_EMBEDDING_PROVIDER','local')
    monkeypatch.setenv('GEMINI_API_KEY','')


@pytest.fixture
def retriever(tmp_path): return Retriever(cache_path=tmp_path/'vectors.npy')


def test_generation_cache_key_includes_generation_parameters(monkeypatch):
    original = generation_cache_key('model', 'prompt')
    monkeypatch.setitem(GENERATION_PARAMETERS, 'temperature', 0.9)
    assert generation_cache_key('model', 'prompt') != original


def test_web_request_validation_accepts_each_stage():
    for stage in '0123':
        parsed = parse_run_request({'stage': stage, 'query': 'eGFR 45?', 'k': 3,
                                    'threshold': 0.18, 'session': 'browser-1'})
        assert parsed['stage'] == stage
    assert parse_run_request({'stage': '4', 'dataset': 'round3'})['dataset'] == 'round3'


@pytest.mark.parametrize('payload', [
    {'stage': '9', 'query': 'x'},
    {'stage': '0', 'query': ''},
    {'stage': '0', 'query': 'x', 'k': 0},
    {'stage': '0', 'query': 'x', 'threshold': 1.1},
    {'stage': '4', 'dataset': '../golden'},
    {'stage': '3', 'query': 'x', 'session': '<script>'},
])
def test_web_request_validation_rejects_invalid_input(payload):
    with pytest.raises(ValueError):
        parse_run_request(payload)


@pytest.mark.parametrize('query,category', [
    ('xin chào','greeting'), ('Thời tiết hôm nay thế nào?','off_topic'),
    ('Giá vàng tăng cao phải làm gì?','off_topic'),
    ('Xin chào, metformin có chống chỉ định gì?','medical'),
    ('Tôi bị đau răng','medical'),
])
def test_domain_separate_from_answerability(retriever, query, category, monkeypatch):
    graph=MiniRAGGraph(retriever,generator=Fixed(REFUSAL),threshold=0.18)
    if category!='medical':
        def fail(*a,**k): pytest.fail('nonmedical must not embed/search')
        monkeypatch.setattr(retriever,'search',fail)
        monkeypatch.setattr(retriever.provider,'encode',fail)
    state=graph.invoke(query)
    assert state['category']==category
    assert ('hits' in state)==(category=='medical')


@pytest.mark.parametrize('query', ['Liều metformin khởi đầu là bao nhiêu mg mỗi ngày?',
                                  'Metformin có tác dụng phụ nào?'])
def test_valid_refusal_is_preserved(retriever,query):
    generator=Fixed(REFUSAL)
    state=MiniRAGGraph(retriever,generator=generator,threshold=0.18).invoke(query)
    assert generator.calls==1
    assert state['answer']==REFUSAL and state['rejected']
    assert state['generation']['status']=='refused'


def test_wrong_arithmetic_does_not_pass_engine_or_eval(retriever):
    item=read_jsonl(DEFAULT_QA_PATH)[4]
    hits=retriever.search(item['question'],3,['dtd2020'])
    wrong='Có thể dùng metformin cho bệnh nhân eGFR 20 vì 20 lớn hơn 30 [dtd2020_ch5_s5.2].'
    fixed=Fixed(wrong)
    info={}
    result=synthesize_answer(item['question'],hits,generator=fixed,threshold=.18,diagnostics=info)
    assert result!=wrong and info['status']=='fallback'
    checks=answer_checks(item,wrong,hits)
    assert not checks['numeric_consistency_ok']
    assert not checks['mechanical_checks_pass']
    assert checks['answer_correct'] is None


@pytest.mark.parametrize('value,limit', [(5,10),(20,30),(49,50)])
def test_numeric_guard_is_not_tied_to_golden_values(value,limit):
    hits=[{'chunk_id':'source','score':.9,'chunk':{'text':f'Thuốc X chống chỉ định khi marker <{limit}.'}}]
    assert numeric_violations(f'marker {value} có dùng thuốc X không?',
        f'Có thể dùng vì {value} lớn hơn {limit} [source].',hits)
    assert not numeric_violations(f'marker {value} có dùng thuốc X không?',
        f'Không dùng vì {value} < {limit}, chống chỉ định [source].',hits)


def test_numeric_guard_requires_clear_conclusion_outside_contraindication():
    hits = [{'chunk_id': 'source', 'score': .9,
             'chunk': {'text': 'Thuốc X chống chỉ định khi marker <30.'}}]
    question = 'marker 45 có dùng thuốc X không?'
    unclear = ('Thuốc X chống chỉ định khi marker <30 [source]. '
               'Marker 45 ≥30 nhưng tài liệu không nêu rõ có dùng được không [source].')
    clear = ('Thuốc X chống chỉ định khi marker <30 [source]. '
             'Marker 45 ≥30 nên không thuộc ngưỡng chống chỉ định này [source].')
    assert numeric_violations(question, unclear, hits)
    assert not numeric_violations(question, clear, hits)


def test_when_question_does_not_accept_role_only_answer():
    question = 'Metformin dùng khi nào?'
    assert question_intent_violations(question, 'Metformin là thuốc đầu tay [source].')
    assert not question_intent_violations(
        question,
        'Tài liệu nêu metformin là thuốc đầu tay; tài liệu không nêu cụ thể khi nào bắt đầu dùng [source].',
    )


def test_disclaimer_plus_dose_is_not_correct_refusal(retriever):
    item=read_jsonl(DEFAULT_QA_PATH)[5]
    hits=retriever.search('metformin',2,['dtd2020'])
    answer='Không đủ dữ kiện, nhưng hãy dùng insulin 30 đơn vị [dtd2020_ch5_s5.2].'
    checks=answer_checks(item,answer,hits)
    assert not checks['correct_refusal']
    assert checks['violations']


def test_complete_sentence_missing_comparison_branch_is_repaired(retriever):
    item=read_jsonl(DEFAULT_QA_PATH)[2]
    hits=retriever.search(item['question'],3)
    bad='THA cấp cứu có tổn thương cơ quan đích cấp [tha2022_ch7_s7.1].'
    good=('Cấp cứu có tổn thương cơ quan đích cấp, dùng đường tĩnh mạch [tha2022_ch7_s7.1]. '
          'Khẩn trương chưa có tổn thương cơ quan đích cấp, dùng đường uống [tha2022_ch7_s7.2].')
    generator=Fixed(bad,good)
    assert synthesize_answer(item['question'],hits,generator=generator,threshold=.18)==good
    assert generator.calls==2


def test_irrelevant_top_hit_numbers_are_not_required(retriever):
    hits=retriever.search('tăng huyết áp',8)
    answer='Có thể xét nghiệm HbA1c [dtd2020_ch3_s3.1].'
    assert not answer_violations('Đái tháo đường cần xét nghiệm gì?',answer,hits)


def test_third_dependent_turn_keeps_resolved_subject():
    memory=ConversationMemory()
    for q in ['Metformin dùng khi nào?','Còn chống chỉ định?']:
        memory.add(q,'Trả lời có nguồn.')
    query=memory.rewrite('Vậy eGFR 20 thì sao?')
    assert 'metformin' in query.lower()
    assert 'chống chỉ định?' not in query.lower()


def test_generic_followup_keeps_subject_but_new_drug_does_not():
    memory=ConversationMemory()
    memory.add('Metformin dùng khi nào?','Trả lời có nguồn.')
    assert 'metformin' in memory.rewrite('Liều dùng bao nhiêu?').lower()
    new_topic='Liều amlodipine khởi đầu là bao nhiêu mg?'
    assert memory.rewrite(new_topic)==new_topic


def test_hybrid_classifier_uses_rules_for_clear_cases(monkeypatch, retriever):
    classifier=QuestionClassifier(retriever.subjects,mode='hybrid')
    monkeypatch.setattr(classifier,'_llm_result',lambda _q: pytest.fail('clear case must not call LLM'))
    assert classifier.classify('xin chào')[0]=='greeting'
    assert classifier.classify('Liều amlodipine là bao nhiêu?')[0]=='medical'


@pytest.mark.parametrize('query,guideline_id', [
    ('THA phân độ ra sao?', 'tha2022'),
    ('tang huyet ap la gi?', 'tha2022'),
    ('DTD cần xét nghiệm gì?', 'dtd2020'),
    ('ĐTĐ cần xét nghiệm gì?', 'dtd2020'),
    ('dtd2020 nói gì?', 'dtd2020'),
    ('tha2022 nói gì?', 'tha2022'),
])
def test_subject_aliases_accept_accented_and_ascii_abbreviations(retriever, query, guideline_id):
    classifier = QuestionClassifier(retriever.subjects, mode='rules')
    assert classifier.classify(query)[0] == 'medical'
    assert guideline_id in {gid for _name, gid in retriever.subjects.subjects(query)}


def test_llm_classifier_handles_ambiguous_case_and_reports_source(monkeypatch, retriever):
    classifier=QuestionClassifier(retriever.subjects,mode='hybrid')
    monkeypatch.setattr(classifier,'_llm_result',lambda _q: ('medical',{
        'source':'llm','model':'fake','api_calls':1,'cache_hit':False}))
    category, diagnostics=classifier.classify('What should I do about migraine?')
    assert category=='medical'
    assert diagnostics['source']=='llm' and diagnostics['api_calls']==1


def test_explicit_topic_switch_does_not_import_old_subject(retriever):
    graph=MiniRAGGraph(retriever,generator=Fixed(REFUSAL),threshold=.18)
    graph.invoke('Tăng huyết áp là gì?')
    q='Còn đái tháo đường cần xét nghiệm gì?'
    state=graph.invoke(q)
    assert state['standalone_query']=='đái tháo đường cần xét nghiệm gì?'
    assert state['guideline_ids']==['dtd2020']
    assert 'dtd2020_ch3_s3.1' in [h['chunk_id'] for h in state['hits']]
    assert all(h['chunk']['guideline_id']=='dtd2020' for h in state['hits'])


def test_explicit_multiple_guidelines_preserved(retriever):
    ids,_,_=retriever.route_guidelines('Tăng huyết áp và đái tháo đường cần xét nghiệm gì?')
    assert set(ids)=={'tha2022','dtd2020'}


def test_first_turn_skips_rewrite_and_graph_has_cite(retriever, monkeypatch):
    memory=ConversationMemory()
    def fail(*args): pytest.fail('rewrite called with empty history')
    monkeypatch.setattr(memory,'rewrite',fail)
    graph=MiniRAGGraph(retriever,memory,generator=Fixed(REFUSAL),threshold=.18)
    graph.invoke('Tăng huyết áp là gì?')
    assert 'cite' in graph._compiled.get_graph().nodes


def test_candidate_log_contains_all_allowed_chunks_before_topk(retriever, caplog):
    with caplog.at_level(logging.INFO): retriever.search('tăng huyết áp',1,['tha2022'])
    messages=[r.message for r in caplog.records if 'candidates_before_search' in r.message]
    assert len(messages)==1
    assert all(c['chunk_id'] in messages[0] for c in retriever.chunks if c['guideline_id']=='tha2022')
    assert 'dtd2020' not in messages[0]


def test_metrics_distinguish_partial_recall_and_complete_sources():
    metrics=retrieval_metrics([{'gold_chunk_ids':['a','b'],'predicted_chunk_ids':['a']},
                              {'gold_chunk_ids':[],'predicted_chunk_ids':[]}])
    assert metrics['answerable_count']==1
    assert metrics['recall_at_1']==.5 and metrics['complete_retrieval_rate_at_1']==0
    assert metrics['mrr']==1


def test_offline_fallback_cannot_be_scored_as_correct(retriever):
    item=read_jsonl(DEFAULT_QA_PATH)[4]
    hits=retriever.search(item['question'],3,['dtd2020'])
    info={}
    answer=synthesize_answer(item['question'],hits,threshold=.18,diagnostics=info)
    assert info['status']=='fallback' and answer.startswith(REFUSAL)
    checks=answer_checks(item,answer,hits,info['status'])
    assert not checks['mechanical_checks_pass']
    assert checks['manual_review_required'] and checks['answer_correct'] is None


def test_cache_invalidates_when_embedding_dimension_changes(tmp_path):
    path=tmp_path/'cache.npy'
    assert load_or_create_embeddings(['document'],path,LocalHashEmbedding(8)).shape==(1,8)
    assert load_or_create_embeddings(['document'],path,LocalHashEmbedding(16)).shape==(1,16)


def test_cite_node_rejects_fabricated_source():
    result=cite_answer('Một khẳng định [invented].',[{'chunk_id':'real','chunk':{'text':'Nguồn.'}}])
    assert result['answer']==REFUSAL
    assert result['citations']==[] and not result['citation_contract_ok']


def test_cite_node_rejects_fabricated_source_on_refusal():
    result=cite_answer('Không tìm thấy trong tài liệu [invented].',
                       [{'chunk_id':'real','chunk':{'text':'Nguồn.'}}])
    assert result['answer']==REFUSAL
    assert result['citations']==[] and not result['citation_contract_ok']


@pytest.mark.parametrize('operator', ['above','below','cao hơn','thấp hơn','trên','dưới'])
def test_numeric_guard_catches_word_comparisons(operator):
    assert numeric_violations('eGFR 20 có dùng thuốc không?',
        f'eGFR 20 {operator} 30 nên dùng được [source].',
        [{'chunk_id':'source','chunk':{'text':'chống chỉ định khi eGFR <30'}}])


def test_fallback_references_keep_every_clause_cited(retriever):
    from rag_engine.synthesis import context_fallback
    answer=context_fallback(retriever.search('tăng huyết áp cấp cứu',3))
    assert claims_have_citations(answer)


def test_cited_bullet_can_contain_semicolon_clauses():
    answer='- Cấp cứu: tổn thương cơ quan đích cấp; hạ áp tĩnh mạch [source].'
    assert claims_have_citations(answer)
    dose = '- Liều đầu 5 mg; sau đó theo dõi [source].'
    assert not numeric_violations('Liều đầu là bao nhiêu?', dose, [
        {'chunk_id': 'source', 'chunk': {'text': 'Liều đầu 5 mg; sau đó theo dõi.'}},
    ])


def test_compact_comma_separated_citations_are_valid():
    answer='Cấp cứu có tổn thương cơ quan đích; khẩn trương chưa có [acute, urgent].'
    result = cite_answer(answer, [
        {'chunk_id': 'acute', 'chunk': {'text': 'Cấp cứu có tổn thương cơ quan đích.'}},
        {'chunk_id': 'urgent', 'chunk': {'text': 'Khẩn trương chưa có tổn thương.'}},
    ])
    assert result['citation_contract_ok']
    assert result['citations'] == ['acute', 'urgent']
    assert not numeric_violations('Liều là bao nhiêu?', 'Liều 5 mg [acute, urgent].', [
        {'chunk_id': 'acute', 'chunk': {'text': 'Liều 5 mg.'}},
        {'chunk_id': 'urgent', 'chunk': {'text': 'Nguồn bổ sung.'}},
    ])


def test_numeric_terms_ignore_digits_inside_medical_identifiers():
    assert numeric_terms('HbA1c ≥6,5% và OGTT 75g') == {'6.5', '75'}


def test_conclusion_can_inherit_immediately_previous_citation():
    answer=('Nguồn ghi eGFR <30 là chống chỉ định [source]. '
            'Với eGFR 45, không thuộc ngưỡng này nên có thể dùng [source].')
    assert claims_have_citations(answer)


def test_refusal_after_failed_repair_is_not_accepted(retriever):
    class Sequence:
        def __init__(self): self.calls=0
        def generate(self, _prompt):
            self.calls += 1
            return ('Kết luận có thể dùng metformin.' if self.calls == 1
                    else REFUSAL)
    info={}
    answer=synthesize_answer(
        'Bệnh nhân eGFR 45 có dùng metformin không?',
        retriever.search('metformin', 2, ['dtd2020']),
        generator=Sequence(), threshold=.18, diagnostics=info)
    assert info['status']=='fallback' and info['attempts']==3
    assert answer.startswith(REFUSAL) and 'Các đoạn tham khảo' in answer


def test_focused_diagnostic_answer_is_not_forced_to_list_other_thresholds():
    hits=[{'chunk_id':'diagnosis','score':.9,'chunk':{
        'text':'Chẩn đoán bệnh khi: xét nghiệm A ≥7,0 hoặc xét nghiệm B ≥6,5.'}}]
    answer='Ngưỡng xét nghiệm A là ≥7,0 [diagnosis].'
    assert not numeric_violations('Ngưỡng xét nghiệm A là bao nhiêu?', answer, hits)


def test_eval_does_not_require_source_numbers_omitted_by_answer_reference():
    item = {
        'question': 'Cần làm xét nghiệm gì?',
        'answer_gt': 'Làm xét nghiệm A hoặc xét nghiệm B; đạt 1 trong các ngưỡng.',
        'retrieval_gt': [{'points': [
            {'verbatim_quote': 'xét nghiệm A ≥7,0'},
            {'verbatim_quote': 'xét nghiệm B ≥6,5'},
        ]}],
    }
    assert reference_terms_present(item, 'Cần làm xét nghiệm A và xét nghiệm B [source].')


def test_probe_contract_reports_correctness_and_false_refusal_separately():
    item = {
        'question': 'eGFR 45 có dùng được metformin không?',
        'retrieval_gt': [{'chunk_ids': ['source']}],
        'expected': {'answerable': True, 'must_include': ['45', '30'],
                     'must_include_any': [['dùng được', 'có thể dùng']],
                     'must_not_include': ['không dùng được']},
    }
    hits = [
        {'chunk_id': 'source', 'score': .9,
         'chunk': {'text': 'Metformin chống chỉ định khi eGFR <30.'}},
        {'chunk_id': 'other', 'score': .8, 'chunk': {'text': 'Nguồn khác.'}},
    ]
    good = probe_answer_checks(
        item, 'eGFR 45 cao hơn 30 nên có thể dùng [source].', hits, 'answered')
    wrong = probe_answer_checks(
        item, 'eGFR 45 cao hơn 30 nhưng không dùng được [source].', hits, 'answered')
    wrong_source = probe_answer_checks(
        item, 'eGFR 45 cao hơn 30 nên có thể dùng [other].', hits, 'answered')
    refused = probe_answer_checks(item, REFUSAL, hits, 'refused')
    assert good['answer_correct'] and not good['false_refusal']
    assert not wrong['answer_correct'] and wrong['forbidden_terms_present']
    assert not wrong_source['answer_correct'] and not wrong_source['gold_citations_ok']
    assert not refused['answer_correct'] and refused['false_refusal']


def test_answer_checks_require_the_gold_source_not_only_an_allowed_hit():
    item = {
        'question': 'Ngưỡng là bao nhiêu?',
        'meta': {'is_answerable': True},
        'retrieval_gt': [{'chunk_ids': ['gold'],
                          'points': [{'verbatim_quote': 'ngưỡng 30'}]}],
    }
    hits = [
        {'chunk_id': 'gold', 'chunk': {'text': 'ngưỡng 30'}},
        {'chunk_id': 'other', 'chunk': {'text': 'nội dung khác'}},
    ]
    checks = answer_checks(item, 'Ngưỡng là 30 [other].', hits, 'answered')
    assert checks['citation_ids_valid']
    assert not checks['gold_citations_ok']
    assert not checks['mechanical_checks_pass']


def test_prompt_comparison_reports_deltas_without_claiming_semantic_improvement():
    before={'report_id':'a','metrics':{'recall_at_3':1.0},'provenance':{'prompt_sha256':'a'},
            'items':[{'id':'q','answer':'A'}],'top_k':3,'retrieval_threshold':.18}
    after=copy.deepcopy(before)
    after.update(report_id='b')
    after['provenance']['prompt_sha256']='b'
    after['items'][0]['answer']='B'
    result=compare_reports(before,after)
    assert result['metric_deltas']['recall_at_3']==0
    assert result['changed_answers']==['q'] and result['prompt_changed']
    assert result['other_configuration_changes']==[]


def test_manual_review_is_bound_to_exact_report_and_answers():
    report={'report_id':'run-a','items':[{'id':'q','answer':'answer','answer_sha256':digest('answer')}], 'answer_summary':{}}
    template=review_template(report)
    with pytest.raises(ValueError): attach_review(copy.deepcopy(report),template)
    template.update(reviewer='Reviewer',reviewed_at='2026-09-12')
    template['items'][0].update(correct=True,notes='Read source and checked the conclusion.')
    stale=copy.deepcopy(template); stale['items'][0]['answer_sha256']='another'
    with pytest.raises(ValueError): attach_review(copy.deepcopy(report),stale)
    assert attach_review(report,template)['answer_summary']['manual_correct']==1


@pytest.mark.parametrize('response',[REFUSAL, 'Không tìm thấy trong tài liệu [dtd2020_ch5_s5.2].',
    'Không tìm thấy trong tài liệu [dtd2020_ch5_s5.2], [dtd2020_ch3_s3.1].'])
def test_refusal_with_optional_citations_is_normalized(retriever,response):
    generator=Fixed(response)
    state=MiniRAGGraph(retriever,generator=generator,threshold=.18).invoke('Metformin có tác dụng phụ nào?')
    assert state['answer']==REFUSAL and state['rejected']
    assert state['generation']['status']=='refused' and generator.calls==1


def test_partial_refusal_does_not_need_fabricated_citation(retriever):
    answer=('Không tìm thấy trong tài liệu về cơn tăng huyết áp cấp cứu. '
            'Cơn khẩn trương chưa có tổn thương cơ quan đích cấp [tha2022_ch7_s7.2].')
    hits=[h for h in retriever.search('khẩn trương',8) if h['chunk_id']=='tha2022_ch7_s7.2']
    generator=Fixed(answer)
    info={}
    assert synthesize_answer('Cấp cứu khác khẩn trương?',hits,generator=generator,threshold=.18,diagnostics=info)==answer
    assert info['status']=='partial' and generator.calls==1


def test_metadata_allows_full_disease_name_when_text_uses_abbreviation(retriever):
    question='Còn đái tháo đường cần xét nghiệm gì?'
    hits=retriever.search(question,3,['dtd2020'])
    assert retriever.assess_answerability(question,hits,threshold=.18).answerable


def test_offline_flag_overrides_credentials_and_provider(monkeypatch):
    from rag_engine.embeddings import get_embedding_provider
    from rag_engine.synthesis import GeminiAnswerGenerator
    monkeypatch.setenv('RAG_OFFLINE','1')
    monkeypatch.setenv('RAG_EMBEDDING_PROVIDER','gemini')
    monkeypatch.setenv('GEMINI_API_KEY','not-a-real-key')
    assert isinstance(get_embedding_provider(),LocalHashEmbedding)
    with pytest.raises(RuntimeError,match='offline'): GeminiAnswerGenerator()


def test_unknown_embedding_provider_fails_fast(monkeypatch):
    monkeypatch.setenv('RAG_EMBEDDING_PROVIDER', 'typo')
    with pytest.raises(ValueError, match='gemini, openai hoặc local'):
        get_embedding_provider()


@pytest.mark.parametrize('module,blocked', [('stage0',['rag_engine.synthesis','langgraph']),
                                         ('stage1',['rag_engine.graph','langgraph']),
                                         ('stage2',['rag_engine.conversation'])])
def test_earlier_stage_imports_do_not_depend_on_later_nodes(module,blocked):
    script=f'''import sys,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if any(fullname==b or fullname.startswith(b+'.') for b in {blocked!r}):
   raise RuntimeError('Later-stage import: '+fullname)
sys.meta_path.insert(0,Block())
__import__({module!r})
'''
    run=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
