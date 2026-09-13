"""LangGraph nodes: rewrite → validate → route → retrieve → synthesize → cite."""
from __future__ import annotations
import logging
from typing import Any, Literal, TypedDict
from langgraph.graph import END, START, StateGraph
from .config import GREETING_RESPONSE, OFF_TOPIC_RESPONSE, RETRIEVAL_THRESHOLD, TOP_K
from .retrieval import Retriever
from .synthesis import synthesize_answer
from .core import QuestionClassifier, cite_answer

LOGGER = logging.getLogger('mini_rag.graph')


class GraphState(TypedDict, total=False):
    question: str
    standalone_query: str
    category: Literal['greeting', 'medical', 'off_topic']
    guideline_ids: list[str]
    route_scores: dict[str, float]
    query_vector: Any
    hits: list[dict]
    answer: str
    citations: list[str]
    rejected: bool
    answerability: dict
    generation: dict
    citation_contract_ok: bool
    classification_source: str
    classification_model: str | None
    classification_api_calls: int


def validate_node(state: GraphState, retriever: Retriever,
                  classifier: QuestionClassifier | None = None) -> dict:
    classifier = classifier or QuestionClassifier(retriever.subjects)
    category, diagnostics = classifier.classify(state['standalone_query'])
    result = {'category': category,
              'classification_source': diagnostics.get('source'),
              'classification_model': diagnostics.get('model'),
              'classification_api_calls': diagnostics.get('api_calls', 0)}
    if category != 'medical':
        result.update(answer=GREETING_RESPONSE if category == 'greeting' else OFF_TOPIC_RESPONSE,
                      rejected=category != 'greeting', citations=[])
    LOGGER.info('validate input=%r -> %s', state['standalone_query'], result)
    return result


def route_node(state: GraphState, retriever: Retriever) -> dict:
    ids, scores, vector = retriever.route_guidelines(state['standalone_query'])
    LOGGER.info('route input=%r -> guidelines=%s scores=%s', state['standalone_query'], ids, scores)
    return {'guideline_ids': ids, 'route_scores': scores, 'query_vector': vector}


class MiniRAGGraph:
    def __init__(self, retriever: Retriever, memory=None, generator=None, *,
                 top_k: int = TOP_K, threshold: float = RETRIEVAL_THRESHOLD,
                 enable_rewrite: bool = True, prompt_variant: str = 'standard',
                 classifier: QuestionClassifier | None = None):
        if top_k < 1:
            raise ValueError('top_k phải >= 1')
        self.retriever, self.generator = retriever, generator
        self.top_k, self.threshold = top_k, threshold
        self.enable_rewrite, self.prompt_variant = enable_rewrite, prompt_variant
        self.classifier = classifier or QuestionClassifier(retriever.subjects)
        self.memory = memory
        if enable_rewrite and self.memory is None:
            from .conversation import ConversationMemory
            self.memory = ConversationMemory(subjects=retriever.subjects)
        elif enable_rewrite:
            # The graph's retriever is the source of truth when callers inject a
            # memory object (especially in tests or with a custom corpus).
            self.memory.subjects = retriever.subjects
        self._compiled = self._build_langgraph()

    def _rewrite(self, state: GraphState) -> dict:
        standalone = self.memory.rewrite(state['question'])
        LOGGER.info('rewrite history=%s input=%r -> standalone=%r',
                    self.memory.history, state['question'], standalone)
        return {'standalone_query': standalone}

    def _retrieve(self, state: GraphState) -> dict:
        hits = self.retriever.search(state['standalone_query'], self.top_k,
                                     state['guideline_ids'], query_vector=state['query_vector'])
        return {'hits': hits}

    def _synthesize(self, state: GraphState) -> dict:
        gate = self.retriever.assess_answerability(state['standalone_query'], state['hits'], self.threshold)
        info = {}
        answer = synthesize_answer(state['question'], state['hits'], threshold=self.threshold,
                                   generator=self.generator, standalone_query=state['standalone_query'],
                                   answerability=gate, diagnostics=info, prompt_variant=self.prompt_variant)
        result = {'answer': answer, 'answerability': gate.as_dict(), 'generation': info,
                  'rejected': info['status'] not in {'answered', 'partial'}}
        LOGGER.info('synthesize hits=%s -> %s', [h['chunk_id'] for h in state['hits']], result)
        return result

    def _cite(self, state: GraphState) -> dict:
        result = cite_answer(state['answer'], state['hits'])
        if not result['citation_contract_ok']:
            result['rejected'] = True
        return result

    def _build_langgraph(self):
        graph = StateGraph(GraphState)
        graph.add_node('validate', lambda s: validate_node(s, self.retriever, self.classifier))
        graph.add_node('route', lambda s: route_node(s, self.retriever))
        graph.add_node('retrieve', self._retrieve)
        graph.add_node('synthesize', self._synthesize)
        graph.add_node('cite', self._cite)
        if self.enable_rewrite:
            graph.add_node('rewrite', self._rewrite)
            graph.add_conditional_edges(START, lambda s: 'rewrite' if self.memory.history else 'validate',
                                        {'rewrite': 'rewrite', 'validate': 'validate'})
            graph.add_edge('rewrite', 'validate')
        else:
            graph.add_edge(START, 'validate')
        graph.add_conditional_edges('validate', lambda s: 'route' if s['category'] == 'medical' else END,
                                    {'route': 'route', END: END})
        graph.add_edge('route', 'retrieve')
        graph.add_edge('retrieve', 'synthesize')
        graph.add_edge('synthesize', 'cite')
        graph.add_edge('cite', END)
        return graph.compile()

    def invoke(self, question: str) -> GraphState:
        if self.enable_rewrite and not self.memory.history:
            LOGGER.info('rewrite skipped: empty history; input=%r', question)
        state = self._compiled.invoke({'question': question, 'standalone_query': question})
        if self.enable_rewrite:
            self.memory.add(question, state['answer'], state['standalone_query'])
        return state
