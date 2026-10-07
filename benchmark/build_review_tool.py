"""Build the standalone human-review tool for CRUD/RGB official labels.

Reads the frozen prepared_v2 files (queue, cases, contexts, corpus, manifest)
and emits a single self-contained HTML page. The page lets a human reviewer,
per case: confirm/edit the official answer, select precise evidence spans
(document_id/start/end/quote) inside the original source text, mark
no-relevant-evidence, or flag anomalies. Progress persists in localStorage;
export writes the protocol-required human_review_confirmed_v1.json.

Frozen inputs are never modified by this script.
"""

import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PREPARED = BASE / "prepared_v2"
OUT = Path(__file__).resolve().parent / "human_review_tool.html"


def load(name: str):
    return json.loads((PREPARED / name).read_text(encoding="utf-8"))


def main() -> None:
    queue = load("human_review_queue.json")
    manifest_bytes = (PREPARED / "manifest.json").read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    crud = load("crud/cases.json")
    rgb = load("rgb/cases.json")
    rgb_ctx = load("rgb/contexts.json")

    def corpus_text(doc_id: str) -> str:
        p = PREPARED / "corpus" / f"{doc_id}.txt"
        if not p.exists():
            return ""
        return p.read_text(encoding="utf-8")

    cases = []
    for c in crud:
        docs = {}
        for ev in c["official_evidence"]:
            did = ev["document_id"]
            if did not in docs:
                docs[did] = corpus_text(did)
        cases.append(
            {
                "track": "crud",
                "case_id": c["case_id"],
                "question": c["question"],
                "reference_answer": c["reference_answer"],
                "task": c["task"],
                "split": c["split"],
                "official_evidence": [
                    {
                        "document_id": ev["document_id"],
                        "start": ev["start"],
                        "end": ev["end"],
                        "quote": docs[ev["document_id"]][ev["start"] : ev["end"]],
                    }
                    for ev in c["official_evidence"]
                ],
                "docs": docs,
            }
        )
    for c in rgb_ctx:
        docs = {did: corpus_text(did) for did in c["context_document_ids"]}
        base = next(b for b in rgb if b["case_id"] == c["base_case_id"])
        cases.append(
            {
                "track": "rgb",
                "case_id": c["case_id"],
                "base_case_id": c["base_case_id"],
                "condition": c["condition"],
                "question": base["question"],
                "reference_answer": base["official_answer"],
                "expected_answerable": c["official_expected_answerable"],
                "noise_rate": c["noise_rate"],
                "relevant_document_ids": c["official_relevant_document_ids"] or [],
                "context_document_ids": c["context_document_ids"],
                "docs": docs,
            }
        )
    queue_map = {q["case_id"]: q for q in queue["cases"]}
    for c in cases:
        c["review_status"] = queue_map[c["case_id"]]["review_status"]
    missing = [c["case_id"] for c in cases if any(not t for t in c["docs"].values())]
    if missing:
        raise SystemExit(f"corpus texts missing for: {missing[:5]}")

    payload = {
        "built_at": "2026-10-05",
        "prepared_manifest_sha256": manifest_sha,
        "cases": cases,
    }
    OUT.write_text(
        TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False)), encoding="utf-8"
    )
    print(
        f"wrote {OUT} with {len(cases)} cases (crud={sum(1 for c in cases if c['track'] == 'crud')}, rgb={sum(1 for c in cases if c['track'] == 'rgb')})"
    )


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>知序 · 官方标签人工核对工具</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--line:#e3e6ea;--ink:#1f2328;--muted:#6a737d;--ok:#1a7f37;--warn:#9a6700;--bad:#cf222e;--accent:#0969da}
*{box-sizing:border-box}
body{margin:0;font-family:"Segoe UI","Microsoft YaHei",sans-serif;background:var(--bg);color:var(--ink);height:100vh;display:flex;flex-direction:column}
header{display:flex;align-items:center;gap:12px;padding:10px 16px;background:var(--card);border-bottom:1px solid var(--line)}
header h1{font-size:16px;margin:0}
header .meta{font-size:12px;color:var(--muted)}
.spacer{flex:1}
button{font:inherit;cursor:pointer;border:1px solid var(--line);background:var(--card);border-radius:6px;padding:5px 12px}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
button.primary:disabled{opacity:.5;cursor:not-allowed}
button.danger{color:var(--bad);border-color:#f0b6bd}
button.small{padding:2px 8px;font-size:12px}
.layout{display:flex;flex:1;overflow:hidden}
.sidebar{width:300px;min-width:300px;background:var(--card);border-right:1px solid var(--line);display:flex;flex-direction:column}
.filters{display:flex;gap:6px;padding:10px;border-bottom:1px solid var(--line);flex-wrap:wrap}
.filters button.active{background:var(--accent);color:#fff;border-color:var(--accent)}
.caselist{flex:1;overflow-y:auto;padding:6px}
.case-item{display:flex;gap:8px;align-items:flex-start;padding:8px;border-radius:6px;cursor:pointer;border-bottom:1px solid #f0f1f2}
.case-item:hover{background:#f0f6ff}
.case-item.sel{background:#ddf0ff}
.case-item .st{flex-shrink:0;font-size:11px;margin-top:2px}
.case-item .q{font-size:13px;line-height:1.4;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.st.pending{color:var(--muted)} .st.done{color:var(--ok)} .st.flag{color:var(--warn)} .st.noev{color:var(--bad)}
.main{flex:1;overflow-y:auto;padding:16px 24px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px;margin-bottom:14px}
.card h3{margin:0 0 8px;font-size:14px}
.question{font-size:15px;font-weight:600;line-height:1.5}
.badge{display:inline-block;font-size:11px;padding:1px 8px;border-radius:10px;border:1px solid var(--line);color:var(--muted);margin-right:6px}
.badge.ans{color:var(--ok);border-color:#b8e0c2}.badge.noans{color:var(--bad);border-color:#f0b6bd}
textarea{width:100%;font:inherit;border:1px solid var(--line);border-radius:6px;padding:8px;resize:vertical}
.doc{display:flex;gap:10px;align-items:flex-start}
.doctext{flex:1;max-height:260px;overflow-y:auto;border:1px solid var(--line);border-radius:6px;padding:10px;background:#fbfcfd;font-size:13px;line-height:1.7;white-space:pre-wrap;user-select:text}
.docmeta{width:220px;min-width:220px;font-size:12px}
.docmeta .did{word-break:break-all;color:var(--muted);font-family:Consolas,monospace;font-size:11px}
.spanchip{display:flex;align-items:center;gap:6px;background:#fff8e5;border:1px solid #f0dc9e;border-radius:6px;padding:4px 8px;margin-top:6px;font-size:12px}
.spanchip .quote{max-height:44px;overflow:hidden;color:#5a5340;flex:1}
.official{background:#eef6ff;border-color:#bcd9f5}
.official .quote{color:#3c5a7a}
.actions{position:sticky;bottom:0;background:var(--bg);padding:10px 0;display:flex;gap:8px;align-items:center}
.hint{font-size:12px;color:var(--muted)}
.note{width:100%;margin-top:8px}
.progress{font-size:12px;color:var(--muted)}
kbd{background:#eef0f2;border:1px solid var(--line);border-radius:4px;padding:0 5px;font-size:11px}
</style>
</head>
<body>
<header>
<h1>知序 · 官方标签人工核对</h1>
<span class="meta">准备 manifest SHA256: <span id="manifestSha"></span> · 构建于 <span id="builtAt"></span></span>
<span class="spacer"></span>
<span class="progress" id="progress"></span>
<button id="importBtn" title="从已导出的确认文件恢复进度">导入进度</button>
<input type="file" id="importFile" accept=".json" style="display:none">
<button id="exportBtn" class="primary">导出确认文件</button>
</header>
<div class="layout">
<div class="sidebar">
  <div class="filters">
    <button data-f="all" class="active">全部</button>
    <button data-f="crud">CRUD</button>
    <button data-f="rgb">RGB</button>
    <button data-f="todo">待核对</button>
  </div>
  <div class="caselist" id="caselist"></div>
</div>
<div class="main" id="main"></div>
</div>
<script>
const DATA = __PAYLOAD__;
const LS_KEY = "zhixu-review-tool-v1";
const LS_NAME = "zhixu-review-name";

function loadState(){
  try{return JSON.parse(localStorage.getItem(LS_KEY))||{};}catch(e){return {}}
}
function saveState(){localStorage.setItem(LS_KEY, JSON.stringify(state));}
let state = loadState();
DATA.cases.forEach(c=>{
  if(!state[c.case_id]) state[c.case_id]={review_status:"pending",confirmed_answer:null,spans:[],notes:""};
});
saveState();

let filter="all", currentIdx=0;
const listEl=document.getElementById("caselist");
const mainEl=document.getElementById("main");

function stLabel(s){
  if(s==="done")return ["done","✔ 已确认"];
  if(s==="noev")return ["noev","✘ 无证据"];
  if(s==="flag")return ["flag","? 存疑"];
  return ["pending","○ 待核对"];
}
function visible(){
  return DATA.cases.map((c,i)=>({c,i})).filter(({c})=>{
    if(filter==="crud")return c.track==="crud";
    if(filter==="rgb")return c.track==="rgb";
    if(filter==="todo")return state[c.case_id].review_status==="pending";
    return true;
  });
}
function renderList(){
  listEl.innerHTML="";
  visible().forEach(({c,i})=>{
    const [cls,label]=stLabel(state[c.case_id].review_status);
    const div=document.createElement("div");
    div.className="case-item"+(i===currentIdx?" sel":"");
    div.innerHTML=`<span class="st ${cls}">${label}</span><span class="q">${esc(c.question)}</span>`;
    div.onclick=()=>{currentIdx=i;renderList();renderMain();};
    listEl.appendChild(div);
  });
  const total=DATA.cases.length, done=DATA.cases.filter(c=>state[c.case_id].review_status!=="pending").length;
  document.getElementById("progress").textContent=`已核对 ${done}/${total}`;
}
function esc(s){return String(s??"").replace(/[&<>"']/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[m]));}
function fmtSpan(s){return `[${s.start},${s.end}) ${s.quote.slice(0,40)}…`;}

function renderMain(){
  const c=DATA.cases[currentIdx];
  const st=state[c.case_id];
  if(!c){mainEl.innerHTML="";return;}
  const officialHtml=(c.track==="crud"&&c.official_evidence&&c.official_evidence.length)
    ?`<div class="card"><h3>官方证据区间（仅参考，请自行核对）</h3>
      ${c.official_evidence.map(ev=>`<div class="spanchip official"><span class="did">${esc(ev.document_id.slice(0,16))}… [${ev.start},${ev.end})</span><span class="quote">${esc(ev.quote.slice(0,120))}</span></div>`).join("")}</div>`:"";
  const rgbMeta=c.track==="rgb"
    ?`<div class="card"><h3>RGB 条件</h3>
      <span class="badge ${c.expected_answerable?"ans":"noans"}">${c.expected_answerable?"官方预期可回答":"官方预期拒答"}</span>
      <span class="badge">噪声率 ${c.noise_rate}</span>
      <span class="badge">相关文档 ${c.relevant_document_ids.length} / 上下文 ${c.context_document_ids.length}</span></div>`:"";
  const spanChips=st.spans.map((s,i)=>`<div class="spanchip"><span class="did">${esc(s.document_id.slice(0,16))}… [${s.start},${s.end})</span><span class="quote">${esc(s.quote.slice(0,120))}</span><button class="small" onclick="removeSpan(${i})">删除</button></div>`).join("");
  const docsHtml=Object.entries(c.docs).map(([did,text])=>`
    <div class="card"><h3>原文 · <code>${esc(did)}</code></h3>
    <div class="doc">
      <div class="doctext" data-doc="${esc(did)}">${esc(text)}</div>
      <div class="docmeta">
        <div class="hint">在左侧原文中用鼠标选中回答所问事实的<b>精确字符区间</b>后松开，自动生成证据条目（可多次选择）。</div>
        ${c.track==="rgb"&&c.relevant_document_ids.includes(did)?'<div class="badge ans" style="margin-top:6px">官方相关</div>':""}
        ${c.track==="rgb"&&!c.relevant_document_ids.includes(did)?'<div class="badge noans" style="margin-top:6px">官方干扰</div>':""}
      </div>
    </div></div>`).join("");
  mainEl.innerHTML=`
    <div class="card"><h3>${esc(c.case_id)} <span class="badge">${c.track.toUpperCase()}</span>${c.task?`<span class="badge">${esc(c.task)}</span>`:""}${c.condition?`<span class="badge">${esc(c.condition)}</span>`:""}</h3>
      <div class="question">${esc(c.question)}</div></div>
    <div class="card"><h3>官方参考答案（核对后可直接确认，或修改为正确版本）</h3>
      <textarea id="answerBox" rows="3">${esc(st.confirmed_answer??c.reference_answer??"")}</textarea></div>
    ${officialHtml}${rgbMeta}
    <div class="card"><h3>人工证据区间</h3>${spanChips||'<div class="hint">尚未选择。若确无相关证据，点下方「标记无相关证据」。</div>'}</div>
    ${docsHtml}
    <div class="card"><h3>备注</h3><textarea id="noteBox" class="note" rows="2" placeholder="如发现官方标签问题，在此记录">${esc(st.notes||"")}</textarea></div>
    <div class="actions">
      <button class="primary" onclick="saveCase('done')">确认答案与证据</button>
      <button class="danger" onclick="saveCase('noev')">标记无相关证据</button>
      <button onclick="saveCase('flag')">标记存疑</button>
      <span class="hint">选择区间：在原文中按住鼠标拖动选择文字，松开即记录。</span>
    </div>`;
  // 选区捕获：仅当选区完全落在单个 doctext 内时记录
  const docs={};
  mainEl.querySelectorAll(".doctext").forEach(el=>{
    el.addEventListener("mouseup",()=>{
      const sel=window.getSelection();
      if(!sel||!sel.rangeCount||sel.isCollapsed)return;
      const r=sel.getRangeAt(0);
      const inEl=el.contains(r.startContainer)&&el.contains(r.endContainer);
      if(!inEl)return;
      const pre=r.cloneRange();
      pre.selectNodeContents(el);pre.setEnd(r.startContainer,r.startOffset);
      const start=pre.toString().length;
      const quote=r.toString().replace(/\s+$/,"");
      const end=start+quote.length;
      if(!quote)return;
      const did=el.dataset.doc;
      const dup=st.spans.some(s=>s.document_id===did&&s.start===start&&s.end===end);
      if(dup)return;
      st.spans.push({document_id:did,start:start,end:end,quote:quote});
      saveState();renderMain();
      sel.removeAllRanges();
    });
  });
}
function removeSpan(i){const st=state[DATA.cases[currentIdx].case_id];st.spans.splice(i,1);saveState();renderMain();}
function saveCase(kind){
  const c=DATA.cases[currentIdx], st=state[c.case_id];
  st.confirmed_answer=document.getElementById("answerBox").value.trim()||null;
  st.notes=document.getElementById("noteBox").value.trim();
  if(kind==="noev"&&st.spans.length&&!confirm("已选择证据区间，仍要标记为无相关证据？"))return;
  if(kind==="noev")st.spans=[];
  if(kind==="done"&&!st.spans.length&&!confirm("未选择任何证据区间，仍按「确认」保存？建议对无证据题使用「标记无相关证据」。"))return;
  st.review_status=kind;
  st.confirmed_at=new Date().toISOString();
  saveState();
  // 前进到下一题：当前题在当前筛选下可能已消失（如“待核对”筛选），此时取其后第一条
  const vis=visible();
  let nextIdx=vis.findIndex(v=>v.i===currentIdx);
  if(nextIdx===-1)nextIdx=vis.findIndex(v=>v.i>currentIdx);
  if(nextIdx===-1)nextIdx=Math.max(vis.length-1,0);
  if(nextIdx>=0&&vis.length)currentIdx=vis[nextIdx].i;
  renderList();renderMain();
}
function exportJson(){
  // 署名决定标签来源，每次导出都必须由本人确认，不能沿用导入的他人署名
  const current=localStorage.getItem(LS_NAME)||"";
  const reviewer=(prompt("审阅者署名（请填本人姓名或账号，此字段代表标签来源，不应填写他人/AI 名称）：",current)||"").trim();
  if(!reviewer){alert("未填写审阅者，导出取消。");return;}
  localStorage.setItem(LS_NAME,reviewer);
  const out={
    reviewer:reviewer,
    exported_at:new Date().toISOString(),
    prepared_manifest_sha256:document.getElementById("manifestSha").textContent,
    source:"review_tool/human_review_tool.html",
    cases:DATA.cases.map(c=>({
      case_id:c.case_id,
      review_status:state[c.case_id].review_status,
      reviewer:state[c.case_id].review_status!=="pending"?reviewer:null,
      confirmed_at:state[c.case_id].confirmed_at||null,
      confirmed_answer:state[c.case_id].confirmed_answer,
      confirmed_evidence_spans:state[c.case_id].review_status==="pending"?null:state[c.case_id].spans,
      notes:state[c.case_id].notes||null,
    })),
  };
  const blob=new Blob([JSON.stringify(out,null,2)],{type:"application/json"});
  const a=document.createElement("a");
  a.href=URL.createObjectURL(blob);
  a.download="human_review_confirmed_v1.json";
  a.click();
  URL.revokeObjectURL(a.href);
  alert(`已导出 ${out.cases.filter(x=>x.review_status!=="pending").length}/${DATA.cases.length} 条。请将文件保存到 prepared_v2/ 目录（文件名保持 human_review_confirmed_v1.json）。`);
}
function importJson(file){
  const reader=new FileReader();
  reader.onload=()=>{
    try{
      const data=JSON.parse(reader.result);
      if(!Array.isArray(data.cases))throw new Error("缺少 cases 数组");
      let n=0;
      data.cases.forEach(c=>{
        if(!state[c.case_id])return;
        if(c.review_status&&c.review_status!=="pending"){
          state[c.case_id].review_status=c.review_status;
          state[c.case_id].confirmed_answer=c.confirmed_answer??null;
          state[c.case_id].spans=c.confirmed_evidence_spans||[];
          state[c.case_id].notes=c.notes||"";
          state[c.case_id].confirmed_at=c.confirmed_at||null;
          n++;
        }
      });
      if(data.reviewer)localStorage.setItem(LS_NAME,data.reviewer);
      saveState();renderList();renderMain();
      alert(`已导入 ${n} 条已核对记录（覆盖同 ID 记录）。`);
    }catch(e){alert("导入失败: "+e.message);}
  };
  reader.readAsText(file,"utf-8");
}
document.getElementById("manifestSha").textContent=DATA.prepared_manifest_sha256.slice(0,16)+"…";
document.getElementById("builtAt").textContent=DATA.built_at;
document.getElementById("exportBtn").onclick=exportJson;
document.getElementById("importBtn").onclick=()=>document.getElementById("importFile").click();
document.getElementById("importFile").onchange=(e)=>{if(e.target.files[0])importJson(e.target.files[0]);e.target.value="";};
document.querySelectorAll(".filters button").forEach(b=>b.onclick=()=>{
  document.querySelectorAll(".filters button").forEach(x=>x.classList.remove("active"));
  b.classList.add("active");filter=b.dataset.f;currentIdx=0;renderList();renderMain();
});
renderList();renderMain();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
