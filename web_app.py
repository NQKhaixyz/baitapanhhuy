"""Giao diện localhost nhỏ để thử từng chặng của Mini RAG Engine.

Chạy: ``uv run python web_app.py --open``
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import re
from threading import Lock
import webbrowser

from rag_engine.config import (
    DEFAULT_QA_PATH,
    RETRIEVAL_THRESHOLD,
    ROOT_DIR,
    TOP_K,
)
from rag_engine.conversation import ConversationMemory
from rag_engine.core import citation_ids
from rag_engine.graph import MiniRAGGraph
from rag_engine.retrieval import Retriever
from rag_engine.synthesis import synthesize_answer


LOGGER = logging.getLogger("mini_rag.web")
MAX_REQUEST_BYTES = 32_768
SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
ROUND3_PATH = ROOT_DIR / "evaluation-dataset" / "round3_aliases.jsonl"


HTML = r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Mini RAG Lab</title>
  <style>
    :root { color-scheme: dark; --bg:#0b1020; --panel:#121a2d; --line:#26324d;
      --text:#edf2ff; --muted:#98a6c6; --blue:#6ea8fe; --cyan:#49d3c6;
      --green:#61d095; --red:#ff7f87; --yellow:#f6c85f; }
    * { box-sizing:border-box } body { margin:0; min-height:100vh; color:var(--text);
      background:radial-gradient(circle at 15% 0,#17264b 0,transparent 38%),var(--bg);
      font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
    button,input,textarea,select { font:inherit } button { cursor:pointer }
    .shell { width:min(1180px,calc(100% - 32px)); margin:28px auto 60px }
    header { display:flex; align-items:flex-end; justify-content:space-between; gap:20px; margin-bottom:22px }
    h1 { margin:0; font-size:clamp(27px,4vw,42px); letter-spacing:-.04em }
    .subtitle,.muted { color:var(--muted) } .subtitle { margin-top:4px }
    .provider { padding:9px 13px; border:1px solid var(--line); border-radius:999px;
      background:#0c1427; color:var(--cyan); white-space:nowrap }
    .stages { display:grid; grid-template-columns:repeat(5,1fr); gap:8px; margin-bottom:14px }
    .stage { border:1px solid var(--line); color:var(--muted); background:#10182a;
      border-radius:13px; padding:11px 8px; text-align:center; transition:.15s }
    .stage:hover { border-color:#40527b; color:var(--text) }
    .stage.active { color:white; border-color:var(--blue); background:#18315f;
      box-shadow:0 0 0 1px #356cc4 inset }
    .stage strong { display:block; font-size:16px } .stage span { font-size:12px }
    .panel { border:1px solid var(--line); background:rgba(18,26,45,.94); border-radius:18px;
      box-shadow:0 18px 50px rgba(0,0,0,.22); padding:20px }
    .stage-title { display:flex; justify-content:space-between; gap:16px; align-items:start }
    .stage-title h2 { margin:0 0 3px; font-size:20px } .stage-title p { margin:0 }
    .controls { display:grid; grid-template-columns:1fr 90px 110px; gap:12px; margin-top:18px }
    label { color:var(--muted); font-size:12px; font-weight:700; letter-spacing:.04em;
      text-transform:uppercase } textarea,input,select { width:100%; margin-top:5px; color:var(--text);
      background:#0a1223; border:1px solid var(--line); border-radius:10px; padding:10px 12px;
      outline:none } textarea:focus,input:focus,select:focus { border-color:var(--blue) }
    textarea { min-height:105px; resize:vertical; text-transform:none; letter-spacing:0; font-weight:500 }
    .query-wrap { grid-column:1/-1 } .dataset-wrap { display:none }
    .actions { display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-top:14px }
    .primary { border:0; color:#07101f; background:linear-gradient(135deg,#79b0ff,#49d3c6);
      border-radius:11px; padding:11px 20px; font-weight:800 }
    .secondary { border:1px solid var(--line); color:var(--text); background:#10192c;
      border-radius:10px; padding:9px 13px }
    button:disabled { opacity:.55; cursor:wait } .samples { margin-left:auto; display:flex; gap:7px; flex-wrap:wrap }
    .sample { border:1px solid var(--line); color:var(--muted); background:transparent;
      border-radius:999px; padding:6px 10px; font-size:12px }
    #status { min-height:24px; margin:15px 2px 0; color:var(--muted) }
    #status.error { color:var(--red) } .spinner { display:inline-block; width:13px; height:13px;
      margin-right:8px; border:2px solid #60708f; border-top-color:var(--cyan); border-radius:50%;
      animation:spin .7s linear infinite; vertical-align:-2px } @keyframes spin { to { transform:rotate(360deg) } }
    #result { display:grid; gap:14px; margin-top:14px }
    .result-card { border:1px solid var(--line); background:#0c1426; border-radius:15px; padding:17px }
    .result-card h3 { margin:0 0 10px; font-size:14px; color:var(--muted); text-transform:uppercase;
      letter-spacing:.06em } .answer { font-size:17px; white-space:pre-wrap; overflow-wrap:anywhere }
    .trace { display:flex; gap:8px; flex-wrap:wrap } .pill { border-radius:999px; padding:6px 10px;
      background:#15213a; border:1px solid #2a3c60; color:#c8d7fa; font-size:13px }
    .pill.good { color:var(--green); border-color:#28654b } .pill.bad { color:var(--red); border-color:#71343b }
    .hits { display:grid; gap:9px } .hit { padding:12px; border-radius:11px; border:1px solid var(--line);
      background:#101a30 } .hit-head { display:flex; justify-content:space-between; gap:12px; font-weight:700 }
    .score { color:var(--cyan); font-variant-numeric:tabular-nums } .hit p { margin:7px 0 0; color:#c2cce3 }
    details { color:var(--muted) } pre { margin:10px 0 0; padding:12px; overflow:auto;
      background:#070d19; border-radius:10px; color:#b9c9e9; font-size:12px }
    table { width:100%; border-collapse:collapse; font-size:13px } th,td { padding:9px 8px;
      border-bottom:1px solid var(--line); text-align:left; vertical-align:top } th { color:var(--muted) }
    .metric-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(145px,1fr)); gap:9px }
    .metric { padding:12px; background:#111c33; border:1px solid var(--line); border-radius:11px }
    .metric b { display:block; margin-top:3px; font-size:20px; color:var(--cyan) }
    @media (max-width:720px) { .shell { width:min(100% - 20px,1180px); margin-top:16px }
      header { align-items:start; flex-direction:column } .stages { grid-template-columns:repeat(5,minmax(82px,1fr)); overflow:auto }
      .controls { grid-template-columns:1fr 1fr } .query-wrap { grid-column:1/-1 }
      .samples { width:100%; margin-left:0 } .panel { padding:15px } }
  </style>
</head>
<body><main class="shell">
  <header><div><h1>Mini RAG Lab</h1><div class="subtitle">Thử từng chặng, nhìn rõ engine quyết định gì.</div></div>
    <div id="provider" class="provider">Đang kết nối engine…</div></header>
  <nav id="stages" class="stages" aria-label="Chọn chặng"></nav>
  <section class="panel">
    <div class="stage-title"><div><h2 id="stageName"></h2><p id="stageDesc" class="muted"></p></div></div>
    <div class="controls">
      <label id="queryWrap" class="query-wrap">Câu hỏi<textarea id="query" spellcheck="false"></textarea></label>
      <label>Top-k<input id="topK" type="number" min="1" max="8" value="3"></label>
      <label>Threshold<input id="threshold" type="number" min="0" max="1" step="0.01" value="0.18"></label>
      <label id="datasetWrap" class="dataset-wrap">Bộ đánh giá<select id="dataset"><option value="golden">Golden · 6 câu</option><option value="round3">Round 3 · 11 câu</option></select></label>
    </div>
    <div class="actions"><button id="run" class="primary">Chạy chặng</button>
      <button id="reset" class="secondary">Xóa hội thoại C3</button><div id="samples" class="samples"></div></div>
    <div id="status" role="status"></div>
  </section>
  <section id="result" aria-live="polite"></section>
</main>
<script>
const stages=[
 {id:'0',name:'C0 · Retriever',short:'Tìm chunk',desc:'Chỉ embedding + cosine. Không gọi model sinh câu trả lời.',sample:'Phân độ tăng huyết áp gồm mấy độ và ngưỡng ra sao?'},
 {id:'1',name:'C1 · Grounded answer',short:'Trả lời',desc:'Retrieve → threshold → sinh đáp án → kiểm tra số và citation.',sample:'Bệnh nhân đái tháo đường có eGFR 45 ml/phút có dùng được metformin không?'},
 {id:'2',name:'C2 · Router',short:'Phân loại',desc:'Validate → route guideline → retrieve → synthesize → cite.',sample:'Liều amlodipine khởi đầu là bao nhiêu mg?'},
 {id:'3',name:'C3 · Conversation',short:'Hội thoại',desc:'Giữ 3 lượt gần nhất và viết lại câu hỏi tiếp theo trước khi tìm.',sample:'Metformin dùng khi nào?'},
 {id:'4',name:'C4 · Eval',short:'Chấm điểm',desc:'Chạy toàn bộ Golden hoặc Round 3 và hiển thị các chỉ số.',sample:''}
];
let current=null;
const savedQueries={};
const session=localStorage.getItem('mini-rag-session') || (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()));
localStorage.setItem('mini-rag-session',session);
const $=id=>document.getElementById(id);
const node=(tag,cls,text)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e};
function selectStage(id){if(current!==null)savedQueries[current]=$('query').value;current=id;const s=stages.find(x=>x.id===id);document.querySelectorAll('.stage').forEach(x=>x.classList.toggle('active',x.dataset.id===id));
 $('stageName').textContent=s.name;$('stageDesc').textContent=s.desc;$('query').value=savedQueries[id]??s.sample;$('query').placeholder=id==='3'?'Nhập từng câu rồi bấm chạy; lịch sử được giữ lại…':'Nhập câu hỏi cần thử…';
 $('queryWrap').style.display=id==='4'?'none':'block';$('datasetWrap').style.display=id==='4'?'block':'none';$('reset').style.display=id==='3'?'inline-block':'none';$('samples').style.display=id==='4'?'none':'flex';$('run').textContent=id==='4'?'Chạy đánh giá':'Chạy chặng';$('result').replaceChildren();$('status').textContent='';}
function addCard(title){const c=node('article','result-card');c.append(node('h3','',title));$('result').append(c);return c}
function pill(text,kind=''){return node('span','pill '+kind,text)}
function renderHits(hits=[]){const c=addCard('Context tìm được');if(!hits.length){c.append(node('div','muted','Không chạy retrieval.'));return}const list=node('div','hits');
 hits.forEach((h,i)=>{const d=node('div','hit'),head=node('div','hit-head');head.append(node('span','',`${i+1}. ${h.chunk_id}`),node('span','score',Number(h.score).toFixed(4)));d.append(head);if(h.section)d.append(node('div','muted',h.section));d.append(node('p','',h.text||''));list.append(d)});c.append(list)}
function renderRun(data){$('result').replaceChildren();const trace=addCard('Luồng xử lý'),line=node('div','trace');
 (data.trace||[]).forEach(x=>line.append(pill(x.text,x.kind||'')));trace.append(line);
 if(data.answer!==undefined&&data.answer!==null){const c=addCard('Câu trả lời');c.append(node('div','answer',data.answer));}
 renderHits(data.hits);const history=data.diagnostics&&data.diagnostics.history_after;if(Array.isArray(history)&&history.length){const c=addCard('Lịch sử C3'),table=node('table'),body=node('tbody');history.forEach((turn,i)=>{const tr=node('tr');tr.append(node('td','',String(i+1)),node('td','',turn.question),node('td','',turn.standalone_query));body.append(tr)});const head=node('tr');['Lượt','Câu gốc','Câu tìm kiếm độc lập'].forEach(x=>head.append(node('th','',x)));const thead=node('thead');thead.append(head);table.append(thead,body);c.append(table)}
 const d=addCard('Chẩn đoán');const details=node('details');details.open=true;details.append(node('summary','', 'Xem state JSON'));details.append(node('pre','',JSON.stringify(data.diagnostics||{},null,2)));d.append(details);}
function metric(c,label,value){const m=node('div','metric');m.append(node('span','muted',label),node('b','',String(value)));c.append(m)}
function renderEval(data){$('result').replaceChildren();const c=addCard('Kết quả đánh giá'),grid=node('div','metric-grid'),m=data.metrics,s=data.summary;
 metric(grid,'Recall@1',pct(m.recall_at_1));metric(grid,'Recall@3',pct(m.recall_at_3));metric(grid,'Complete@3',pct(m.complete_retrieval_rate_at_3));metric(grid,'MRR',pct(m.mrr));metric(grid,'Mechanical',`${s.mechanical_pass}/${s.answerable_count}`);
 if(s.probe_metrics&&s.probe_metrics.answer_correct_rate!==null){metric(grid,'Probe đúng',pct(s.probe_metrics.answer_correct_rate));metric(grid,'Từ chối nhầm',pct(s.probe_metrics.false_refusal_rate));}c.append(grid,node('p','muted',`Artifact: ${data.artifact}`));
 const t=addCard('Từng câu'),table=node('table'),head=node('tr');['ID','Top chunks','Generation','Kết quả'].forEach(x=>head.append(node('th','',x)));const thead=node('thead');thead.append(head);table.append(thead);const body=node('tbody');
 data.items.forEach(i=>{const tr=node('tr');tr.append(node('td','',i.id),node('td','',i.predicted_chunk_ids.join(', ')||'—'),node('td','',`${i.generation.status} · API ${i.generation.api_calls} · cache ${i.generation.cache_hits}`),node('td','',i.result));body.append(tr)});table.append(body);t.append(table)}
const pct=v=>v===null||v===undefined?'—':`${(Number(v)*100).toFixed(1)}%`;
async function post(path,payload){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const data=await r.json();if(!r.ok)throw new Error(data.error||`HTTP ${r.status}`);return data}
async function run(){const query=$('query').value.trim();if(current!=='4'&&!query){$('status').className='error';$('status').textContent='Hãy nhập câu hỏi.';return}
 $('run').disabled=true;$('status').className='';$('status').replaceChildren(node('span','spinner'),document.createTextNode(current==='4'?'Đang chạy cả bộ đánh giá…':'Engine đang xử lý…'));
 try{const data=await post('/api/run',{stage:current,query,k:Number($('topK').value),threshold:Number($('threshold').value),dataset:$('dataset').value,session});current==='4'?renderEval(data):renderRun(data);$('status').textContent=`Xong trong ${data.elapsed_ms} ms.`;if(data.provider)$('provider').textContent=`${data.provider.name} · ${data.provider.model||'local'} · ${data.provider.dimension} chiều`;}
 catch(e){$('status').className='error';$('status').textContent=e.message}finally{$('run').disabled=false}}
async function reset(){try{await post('/api/reset',{session});$('status').className='';$('status').textContent='Đã xóa lịch sử hội thoại C3.';$('result').replaceChildren()}catch(e){$('status').className='error';$('status').textContent=e.message}}
stages.forEach(s=>{const b=node('button','stage');b.dataset.id=s.id;b.append(node('strong','',s.name.split(' · ')[0]),node('span','',s.short));b.onclick=()=>selectStage(s.id);$('stages').append(b)});
['eGFR 20 có dùng được metformin không?','Thế phân độ ra sao?','xin chào'].forEach(x=>{const b=node('button','sample',x);b.onclick=()=>{$('query').value=x};$('samples').append(b)});
$('run').onclick=run;$('reset').onclick=reset;$('query').addEventListener('keydown',e=>{if(e.ctrlKey&&e.key==='Enter')run()});
fetch('/api/health').then(r=>r.json()).then(d=>{$('provider').textContent=`${d.provider.name} · ${d.provider.model||'local'} · ${d.provider.dimension} chiều`}).catch(()=>{$('provider').textContent='Không kết nối được engine'});
selectStage('0');
</script></body></html>"""


def parse_run_request(payload: object) -> dict:
    """Validate the small public HTTP contract before touching the engine."""
    if not isinstance(payload, dict):
        raise ValueError("Body phải là JSON object.")
    stage = str(payload.get("stage", ""))
    if stage not in {"0", "1", "2", "3", "4"}:
        raise ValueError("Chặng phải nằm trong 0–4.")
    query = str(payload.get("query", "")).strip()
    if stage != "4" and not query:
        raise ValueError("Cần nhập câu hỏi.")
    if len(query) > 4_000:
        raise ValueError("Câu hỏi dài tối đa 4.000 ký tự.")
    try:
        k = int(payload.get("k", TOP_K))
        threshold = float(payload.get("threshold", RETRIEVAL_THRESHOLD))
    except (TypeError, ValueError) as exc:
        raise ValueError("Top-k hoặc threshold không hợp lệ.") from exc
    if not 1 <= k <= 8:
        raise ValueError("Top-k phải nằm trong 1–8.")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold phải nằm trong 0–1.")
    dataset = str(payload.get("dataset", "golden"))
    if dataset not in {"golden", "round3"}:
        raise ValueError("Dataset phải là golden hoặc round3.")
    session = str(payload.get("session", "default"))
    if not SESSION_RE.fullmatch(session):
        raise ValueError("Session không hợp lệ.")
    return {"stage": stage, "query": query, "k": k, "threshold": threshold,
            "dataset": dataset, "session": session}


class EngineService:
    """One shared retriever plus isolated in-memory C3 conversations."""

    def __init__(self) -> None:
        self.retriever = Retriever()
        self.memories: dict[str, ConversationMemory] = {}
        self.lock = Lock()

    def provider_info(self) -> dict:
        provider = self.retriever.provider
        return {"name": type(provider).__name__, "model": getattr(provider, "model", None),
                "dimension": int(self.retriever.vectors.shape[1])}

    @staticmethod
    def _hits(hits: list[dict]) -> list[dict]:
        return [{"chunk_id": hit["chunk_id"], "score": hit["score"],
                 "section": str(hit["chunk"].get("section_path", "")),
                 "text": str(hit["chunk"].get("text", ""))} for hit in hits]

    def reset(self, session: str) -> None:
        with self.lock:
            self.memories.pop(session, None)

    def run(self, request: dict) -> dict:
        started = datetime.now(UTC)
        with self.lock:
            result = self._run_locked(request)
        result["provider"] = self.provider_info()
        result["elapsed_ms"] = round((datetime.now(UTC) - started).total_seconds() * 1000)
        return result

    def _run_locked(self, request: dict) -> dict:
        stage, query = request["stage"], request["query"]
        k, threshold = request["k"], request["threshold"]
        if stage == "4":
            return self._run_eval(request["dataset"], k, threshold)
        if stage == "0":
            hits = self.retriever.search(query, k)
            return {"answer": None, "hits": self._hits(hits),
                    "trace": [{"text": "C0 retrieve", "kind": "good"}],
                    "diagnostics": {"query": query, "top_k": k}}
        if stage == "1":
            hits = self.retriever.search(query, k)
            gate = self.retriever.assess_answerability(query, hits, threshold)
            generation: dict = {}
            answer = synthesize_answer(query, hits, threshold=threshold,
                                       answerability=gate, diagnostics=generation)
            return {"answer": answer, "hits": self._hits(hits),
                    "trace": [{"text": "C0 retrieve", "kind": "good"},
                              {"text": "C1 synthesize", "kind": "good" if generation.get("status") in {"answered", "partial"} else "bad"}],
                    "diagnostics": {"answerability": gate.as_dict(), "generation": generation,
                                    "citations": citation_ids(answer)}}
        if stage == "2":
            graph = MiniRAGGraph(self.retriever, top_k=k, threshold=threshold,
                                 enable_rewrite=False)
            state = graph.invoke(query)
            return self._graph_result(state, include_rewrite=False)
        memory = self.memories.setdefault(
            request["session"], ConversationMemory(subjects=self.retriever.subjects)
        )
        history_before = list(memory.history)
        graph = MiniRAGGraph(self.retriever, memory, top_k=k, threshold=threshold)
        state = graph.invoke(query)
        result = self._graph_result(state, include_rewrite=True)
        result["diagnostics"]["history_before"] = history_before
        result["diagnostics"]["history_after"] = list(memory.history)
        return result

    def _graph_result(self, state: dict, *, include_rewrite: bool) -> dict:
        trace = []
        if include_rewrite:
            changed = state.get("standalone_query") != state.get("question")
            trace.append({"text": "C3 rewrite" + (" · đã viết lại" if changed else " · giữ nguyên"),
                          "kind": "good"})
        category = state.get("category", "unknown")
        trace.append({"text": f"C2 validate · {category}",
                      "kind": "bad" if category == "off_topic" else "good"})
        if state.get("guideline_ids"):
            trace.append({"text": "C2 route · " + ", ".join(state["guideline_ids"]), "kind": "good"})
            trace.append({"text": "C0 retrieve", "kind": "good"})
            trace.append({"text": "C1 " + state.get("generation", {}).get("status", "synthesize"),
                          "kind": "good" if not state.get("rejected") else "bad"})
        diagnostics = {key: state.get(key) for key in (
            "question", "standalone_query", "category", "classification_source",
            "classification_model", "guideline_ids", "route_scores", "answerability",
            "generation", "citations", "citation_contract_ok", "rejected"
        ) if key in state}
        return {"answer": state.get("answer"), "hits": self._hits(state.get("hits", [])),
                "trace": trace, "diagnostics": diagnostics}

    def _run_eval(self, dataset: str, k: int, threshold: float) -> dict:
        from eval import run_eval

        qa_path = DEFAULT_QA_PATH if dataset == "golden" else ROUND3_PATH
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output = ROOT_DIR / "artifacts" / f"web_eval_{dataset}_{stamp}.json"
        report = run_eval(output, top_k=k, threshold=threshold, qa_path=qa_path)
        items = []
        for item in report["items"]:
            probe = item.get("probe_checks")
            checks = item["checks"]
            passed = probe.get("answer_correct") if probe else (
                checks.get("mechanical_checks_pass") if checks.get("answerable")
                else checks.get("correct_refusal")
            )
            items.append({"id": item["id"],
                          "predicted_chunk_ids": item["predicted_chunk_ids"],
                          "generation": item["generation"],
                          "result": "Đạt" if passed else "Cần xem lại"})
        return {"metrics": report["metrics"], "summary": report["answer_summary"],
                "items": items, "artifact": str(output.relative_to(ROOT_DIR))}


class AppHandler(BaseHTTPRequestHandler):
    server_version = "MiniRAGWeb/1.0"

    @property
    def service(self) -> EngineService:
        return self.server.service  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: object) -> None:
        LOGGER.info("%s - %s", self.client_address[0], fmt % args)

    def _headers(self, status: HTTPStatus, content_type: str, length: int) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'")
        self.end_headers()

    def _send_json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(data))
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            data = HTML.encode("utf-8")
            self._headers(HTTPStatus.OK, "text/html; charset=utf-8", len(data))
            self.wfile.write(data)
        elif self.path == "/api/health":
            self._send_json({"ok": True, "provider": self.service.provider_info()})
        else:
            self._send_json({"error": "Không tìm thấy đường dẫn."}, HTTPStatus.NOT_FOUND)

    def _read_payload(self) -> object:
        if self.headers.get_content_type() != "application/json":
            raise ValueError("Content-Type phải là application/json.")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Content-Length không hợp lệ.") from exc
        if not 0 < length <= MAX_REQUEST_BYTES:
            raise ValueError("Request rỗng hoặc quá lớn.")
        try:
            return json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("JSON không hợp lệ.") from exc

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._read_payload()
            if self.path == "/api/run":
                self._send_json(self.service.run(parse_run_request(payload)))
            elif self.path == "/api/reset":
                if not isinstance(payload, dict):
                    raise ValueError("Body phải là JSON object.")
                session = str(payload.get("session", ""))
                if not SESSION_RE.fullmatch(session):
                    raise ValueError("Session không hợp lệ.")
                self.service.reset(session)
                self._send_json({"ok": True})
            else:
                self._send_json({"error": "Không tìm thấy đường dẫn."}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # API errors are logged locally without leaking internals.
            LOGGER.exception("Web request failed")
            self._send_json({"error": f"Engine lỗi: {type(exc).__name__}. Xem terminal chạy web."},
                            HTTPStatus.INTERNAL_SERVER_ERROR)


class MiniRAGServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], service: EngineService):
        super().__init__(address, AppHandler)
        self.service = service


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="Mở trình duyệt sau khi server sẵn sàng")
    parser.add_argument("--verbose", action="store_true", help="Hiện log chi tiết của từng node")
    args = parser.parse_args()
    if not 1 <= args.port <= 65_535:
        parser.error("port phải nằm trong 1–65535")
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s | %(message)s",
    )
    service = EngineService()
    server = MiniRAGServer((args.host, args.port), service)
    url = f"http://{args.host}:{args.port}"
    print(f"Mini RAG Lab đang chạy tại {url}", flush=True)
    print("Nhấn Ctrl+C để dừng.", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng Mini RAG Lab.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
